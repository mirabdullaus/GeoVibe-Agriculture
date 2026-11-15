
import logging
import math
import os
import re
from calendar import monthrange
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import ee
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from scipy.stats import kendalltau, theilslopes

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

app = FastAPI(title="GeoVibe Agriculture")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("geovibe")

EE_PROJECT = os.getenv("EE_PROJECT", "").strip()
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "").strip()
MAPBOX_STYLE = os.getenv(
    "MAPBOX_STYLE",
    "mapbox://styles/mapbox/satellite-streets-v12"
).strip()

S2 = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
L8 = "LANDSAT/LC08/C02/T1_L2"
L9 = "LANDSAT/LC09/C02/T1_L2"

INDICES = ["NDVI", "NDMI", "EVI"]
BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR"] + INDICES
CLEAR_THRESHOLD = 0.65
SCENE_COVERAGE = 70.0
RGB_COVERAGE = 95.0
PERIOD_COVERAGE = 70.0
MAX_SCENES = 1600

PALETTES = {
    "NDVI": ["#bc573b", "#e9af4b", "#ece77c", "#92c75d", "#217646"],
    "NDMI": ["#b95c42", "#e9ce96", "#a4d6c1", "#51a9c4", "#205f9b"],
    "EVI": ["#b75c40", "#e9bd66", "#a3cd66", "#217047"],
    "CHANGE": ["#d65842", "#f4b36d", "#f1efdc", "#a5d58d", "#2c8255"]
}
RANGES = {
    "NDVI": (0, 0.85),
    "NDMI": (-0.45, 0.55),
    "EVI": (0, 0.8),
    "CHANGE": (-0.3, 0.3)
}
EE_READY = False


class AnalysisRequest(BaseModel):
    geometry: dict
    start: date
    end: date


class SceneRequest(BaseModel):
    geometry: dict
    asset: str
    layer: str = "RGB"


class ChangeRequest(BaseModel):
    geometry: dict
    period_from: str
    period_to: str


def initialize_ee():
    global EE_READY
    if not EE_READY:
        if not EE_PROJECT:
            raise RuntimeError("EE_PROJECT is missing from .env")
        ee.Initialize(project=EE_PROJECT)
        EE_READY = True


def validate_geometry(data):
    if data.get("type") != "Polygon":
        raise ValueError("Draw one polygon.")
    rings = data.get("coordinates")
    if not isinstance(rings, list) or len(rings) != 1:
        raise ValueError("A polygon without holes is required.")
    ring = rings[0]
    if not isinstance(ring, list) or not 4 <= len(ring) <= 101:
        raise ValueError("Invalid polygon vertices.")
    if ring[0] != ring[-1]:
        raise ValueError("Polygon must be closed.")

    for pt in ring:
        if not isinstance(pt, list) or len(pt) != 2:
            raise ValueError("Invalid coordinates.")
        if any(
            isinstance(v, bool) or
            not isinstance(v, (int, float)) or
            not math.isfinite(v)
            for v in pt
        ):
            raise ValueError("Invalid coordinates.")
        if not (-180 <= pt[0] <= 180 and -90 <= pt[1] <= 90):
            raise ValueError("Coordinates outside geographic range.")

    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    if max(xs) - min(xs) > .12 or max(ys) - min(ys) > .12:
        raise ValueError("Please draw a smaller agricultural field.")
    signed = sum(
        ring[i][0] * ring[i + 1][1] -
        ring[i + 1][0] * ring[i][1]
        for i in range(len(ring) - 1)
    )
    if abs(signed) < 1e-10:
        raise ValueError("Polygon is too small.")

    return ee.Geometry.Polygon(
        rings, proj="EPSG:4326", geodesic=False
    )


def empty_image():
    return (ee.Image.constant([0] * len(BANDS))
            .rename(BANDS)
            .updateMask(ee.Image.constant(0)))


def add_indices(blue, green, red, nir, swir):
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    ndmi = nir.subtract(swir).divide(nir.add(swir)).rename("NDMI")
    evi = nir.expression(
        "2.5*(n-r)/(n+6*r-7.5*b+1)",
        {"n": nir, "r": red, "b": blue}
    ).rename("EVI")
    image = ee.Image.cat([
        blue.rename("BLUE"), green.rename("GREEN"),
        red.rename("RED"), nir.rename("NIR"),
        swir.rename("SWIR"), ndvi, ndmi, evi
    ])
    common_mask = image.mask().reduce(ee.Reducer.min())
    return image.updateMask(common_mask)


def process_sentinel(raw):
    image = ee.Image(raw)
    scl = image.select("SCL")
    mask = (
        scl.neq(0).And(scl.neq(1))
        .And(scl.neq(3)).And(scl.neq(8))
        .And(scl.neq(9)).And(scl.neq(10))
        .And(scl.neq(11))
        .And(image.select("cs_cdf").gte(CLEAR_THRESHOLD))
    )
    r = image.select(
        ["B2", "B3", "B4", "B8", "B11"]
    ).multiply(.0001)
    blue, green, red = r.select("B2"), r.select("B3"), r.select("B4")
    nir, swir = r.select("B8"), r.select("B11")
    for band in (blue, green, red, nir, swir):
        mask = mask.And(band.gt(0))
    return (
        add_indices(blue, green, red, nir, swir)
        .updateMask(mask)
        .copyProperties(image, ["system:time_start"])
        .set({
            "sensor": "Sentinel-2",
            "asset": ee.String(S2).cat("/").cat(
                ee.String(image.get("system:index"))
            )
        })
    )


def process_landsat(raw, collection_id, sensor):
    image = ee.Image(raw)
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(63).eq(0)
        .And(qa.rightShift(8).bitwiseAnd(3).lte(1))
        .And(qa.rightShift(10).bitwiseAnd(3).lte(1))
        .And(qa.rightShift(14).bitwiseAnd(3).lte(1))
        .And(image.select("QA_RADSAT").eq(0))
    )
    r = image.select([
        "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6"
    ]).multiply(.0000275).add(-.2)
    blue, green = r.select("SR_B2"), r.select("SR_B3")
    red, nir = r.select("SR_B4"), r.select("SR_B5")
    swir = r.select("SR_B6")
    for band in (blue, green, red, nir, swir):
        mask = mask.And(band.gt(0))
    return (
        add_indices(blue, green, red, nir, swir)
        .updateMask(mask)
        .copyProperties(image, ["system:time_start"])
        .set({
            "sensor": sensor,
            "asset": ee.String(collection_id).cat("/").cat(
                ee.String(image.get("system:index"))
            )
        })
    )


def get_collections(geom, start, end):
    first = start.isoformat()
    last = (end + timedelta(days=1)).isoformat()

    sentinel = (
        ee.ImageCollection(S2)
        .filterBounds(geom).filterDate(first, last)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 95))
        .linkCollection(ee.ImageCollection(CLOUD_SCORE), ["cs_cdf"])
        .map(process_sentinel)
    )
    l8 = (ee.ImageCollection(L8)
          .filterBounds(geom).filterDate(first, last)
          .map(lambda im: process_landsat(im, L8, "Landsat 8")))
    l9 = (ee.ImageCollection(L9)
          .filterBounds(geom).filterDate(first, last)
          .map(lambda im: process_landsat(im, L9, "Landsat 9")))
    return sentinel, l8, l9


def median_or_empty(collection):
    return ee.Image(ee.Algorithms.If(
        collection.size().gt(0),
        collection.median(),
        empty_image()
    )).select(BANDS)


def combined_image(sentinel, landsat):
    return ee.Image(median_or_empty(sentinel).unmask(
        median_or_empty(landsat), sameFootprint=False
    ))


def region_stats(image, geom):
    image = ee.Image(image)
    means = image.select(INDICES).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom,
        scale=30, maxPixels=2000000, tileScale=4
    )
    valid = (
        ee.Image.pixelArea()
        .updateMask(image.select("NDVI").mask())
        .rename("valid_area")
        .reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geom,
            scale=30, maxPixels=2000000, tileScale=4
        )
    )
    return ee.Dictionary(means).combine(valid)


def coverage_pct(valid, area):
    return round(min(100, max(0, 100 * valid / area)), 2) if area > 0 else 0


def parse_stats(props, area, is_scene=False):
    row = {
        "date": str(props.get("date") or ""),
        "coverage_pct": coverage_pct(float(props.get("valid_area") or 0), area)
    }
    for key in INDICES:
        v = props.get(key)
        row[key] = (
            round(float(v), 5)
            if isinstance(v, (int, float)) and math.isfinite(v)
            else None
        )
    if is_scene:
        row.update({
            "timestamp": int(props.get("timestamp") or 0),
            "asset": str(props.get("asset") or ""),
            "sensor": str(props.get("sensor") or "")
        })
    return row


def scene_feature(raw, geom):
    image = ee.Image(raw)
    properties = ee.Dictionary({
        "date": ee.Date(image.get("system:time_start")).format("YYYY-MM-dd"),
        "timestamp": image.get("system:time_start"),
        "asset": image.get("asset"),
        "sensor": image.get("sensor")
    })
    return ee.Feature(None, region_stats(image, geom).combine(properties))


def scene_rows(collection, count, geom, area):
    items = collection.sort("system:time_start").toList(count)
    output = []
    for offset in range(0, count, 50):
        batch = items.slice(offset, min(offset + 50, count))
        features = ee.FeatureCollection(
            batch.map(lambda item: scene_feature(ee.Image(item), geom))
        ).getInfo()["features"]
        output.extend(
            parse_stats(f["properties"], area, is_scene=True)
            for f in features
        )
    return sorted(output, key=lambda x: (
        x["timestamp"], x["sensor"], x["asset"]
    ))


def selected_periods(start, end):
    if (end - start).days <= 730:
        ids = []
        year, month = start.year, start.month
        while (year, month) <= (end.year, end.month):
            first = date(year, month, 1)
            last = date(year, month, monthrange(year, month)[1])
            if first >= start and last <= end:
                ids.append(f"{year}-{month:02d}")
            month += 1
            if month == 13:
                year, month = year + 1, 1
        return "monthly", ids

    return "seasonal", [
        str(year) for year in range(start.year, end.year + 1)
        if date(year, 5, 1) >= start and date(year, 8, 31) <= end
    ]


def period_dates(identifier):
    if re.fullmatch(r"\d{4}-\d{2}", identifier):
        year, month = map(int, identifier.split("-"))
        if not 1 <= month <= 12:
            raise ValueError("Invalid month.")
        first = date(year, month, 1)
        last = date(year, month, monthrange(year, month)[1])
    elif re.fullmatch(r"\d{4}", identifier):
        year = int(identifier)
        first, last = date(year, 5, 1), date(year, 8, 31)
    else:
        raise ValueError("Invalid period identifier.")
    if first < date(2017, 3, 1) or first > date.today():
        raise ValueError("Period is outside the supported range.")
    return first.isoformat(), (last + timedelta(days=1)).isoformat()


def period_image(sentinel, landsat, identifier):
    first, last = period_dates(identifier)
    return combined_image(
        sentinel.filterDate(first, last),
        landsat.filterDate(first, last)
    )


def calculate_periods(sentinel, landsat, ids, geom, area):
    if not ids:
        return [], {}
    images = {}
    features = []
    for identifier in ids:
        image = period_image(sentinel, landsat, identifier)
        images[identifier] = image
        feature = ee.Feature(None, region_stats(image, geom).combine(
            ee.Dictionary({"date": identifier})
        ))
        features.append(feature)
    result = ee.FeatureCollection(features).getInfo()["features"]
    rows = [parse_stats(f["properties"], area) for f in result]
    return rows, images


def reference_periods(mode, end, selected_ids):
    latest = selected_ids[-1] if selected_ids else (
        f"{end.year}-{end.month:02d}" if mode == "monthly" else str(end.year)
    )
    latest_year = int(latest[:4])
    if mode == "monthly":
        return [
            f"{year}-{latest[-2:]}"
            for year in range(max(2017, latest_year - 5), latest_year + 1)
        ]
    return [
        str(year)
        for year in range(max(2017, latest_year - 5), latest_year + 1)
    ]


def numeric_time(identifier):
    if len(identifier) == 4:
        return float(identifier)
    year, month = map(int, identifier.split("-"))
    return year + (month - .5) / 12


def historical_trend(rows, variable):
    valid = [
        r for r in rows
        if r.get("coverage_pct", 100) >= PERIOD_COVERAGE
        and isinstance(r.get(variable), (int, float))
    ]
    if len(valid) < 2:
        return {"status": "insufficient", "n": len(valid)}

    x = [numeric_time(r["date"]) for r in valid]
    y = [float(r[variable]) for r in valid]
    result = theilslopes(y, x)
    out = {
        "status": "descriptive",
        "n": len(valid),
        "slope_year": round(float(result.slope), 5),
        "p_value": None,
        "ci_low": None,
        "ci_high": None
    }
    if len(valid) >= 5 and len(set(y)) >= 2:
        tau, p = kendalltau(x, y)
        out.update({
            "status": "exploratory",
            "p_value": round(float(p), 5) if math.isfinite(float(p)) else None,
            "tau": round(float(tau), 5) if math.isfinite(float(tau)) else None,
            "ci_low": round(float(result.low_slope), 5),
            "ci_high": round(float(result.high_slope), 5)
        })
    return out


def historical_score(rows):
    valid = [
        r for r in rows
        if r["coverage_pct"] >= PERIOD_COVERAGE
        and all(isinstance(r.get(index), (int, float)) for index in INDICES)
    ]
    if len(valid) < 4:
        return {
            "value": None,
            "status": "insufficient_baseline",
            "reference_years": max(0, len(valid) - 1)
        }
    current, baseline = valid[-1], valid[:-1]
    weights = {"NDVI": .5, "NDMI": .3, "EVI": .2}
    score = 0
    for index, weight in weights.items():
        below = sum(r[index] < current[index] for r in baseline)
        equal = sum(r[index] == current[index] for r in baseline)
        score += weight * (below + .5 * equal) / len(baseline)

    return {
        "value": round(score * 100),
        "status": "historical_relative",
        "reference_years": len(baseline),
        "reference_period": current["date"],
        "method": (
            "Weighted percentile of NDVI (50%), NDMI (30%) and EVI (20%) "
            "against earlier matching periods. Experimental and unvalidated."
        )
    }


@lru_cache(maxsize=128)
def nasa_power(latitude, longitude, first_year, last_year):
    response = requests.get(
        "https://power.larc.nasa.gov/api/temporal/monthly/point",
        params={
            "parameters": "PRECTOTCORR",
            "community": "AG",
            "latitude": latitude,
            "longitude": longitude,
            "start": first_year,
            "end": last_year,
            "format": "JSON"
        },
        timeout=90
    )
    response.raise_for_status()
    payload = response.json()
    raw = payload["properties"]["parameter"]["PRECTOTCORR"]
    units = payload.get("parameters", {}).get("PRECTOTCORR", {}).get(
        "units", "mm/day"
    )
    # POWER monthly PRECTOTCORR is monthly mean daily rainfall (mm/day).
    # Convert to monthly accumulated rainfall in millimetres.
    values = {}
    for key, value in raw.items():
        if len(key) != 6 or not key.isdigit():
            continue
        year, month = int(key[:4]), int(key[4:])
        if not 1 <= month <= 12:
            continue
        v = float(value)
        if math.isfinite(v) and v > -900:
            values[key] = round(v * monthrange(year, month)[1], 2)
    return values


def precipitation_data(geom, start, end, identifiers):
    centre = geom.centroid(maxError=10).coordinates().getInfo()
    lon, lat = round(float(centre[0]), 4), round(float(centre[1]), 4)

    years = [start.year, end.year] + [int(p[:4]) for p in identifiers]
    raw = nasa_power(lat, lon, min(years), max(years))

    # Entire selected period: monthly chart, separate from trend references.
    monthly = []
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            first = date(year, month, 1)
            last = date(year, month, monthrange(year, month)[1])
            key = f"{year}{month:02d}"
            if first >= start and last <= end and key in raw:
                monthly.append({
                    "date": f"{year}-{month:02d}",
                    "mm": raw[key]
                })

    # Historical reference periods: Trends and condition context.
    periods = []
    for identifier in sorted(set(identifiers)):
        if len(identifier) == 7:
            key = identifier.replace("-", "")
            if key not in raw:
                continue
            total = raw[key]
        else:
            keys = [f"{identifier}{month:02d}" for month in range(5, 9)]
            if not all(k in raw for k in keys):
                continue
            total = sum(raw[k] for k in keys)
        periods.append({
            "date": identifier,
            "mm": round(total, 2),
            "coverage_pct": 100
        })

    return {
        "monthly": monthly,
        "periods": periods,
        "trend": historical_trend(periods, "mm"),
        "location": {"latitude": lat, "longitude": lon},
        "units": "mm",
        "source": "NASA POWER"
    }


def visualize(image, layer):
    image = ee.Image(image)
    if layer == "RGB":
        return ee.Image(image.visualize(
            bands=["RED", "GREEN", "BLUE"],
            min=0, max=.30, gamma=1.1
        ))
    if layer not in RANGES:
        raise ValueError("Unsupported layer.")
    lo, hi = RANGES[layer]
    return ee.Image(image.visualize(
        bands=[layer], min=lo, max=hi, palette=PALETTES[layer]
    ))


def tile_url(image, layer, geom):
    image = ee.Image(image).clip(geom)
    return visualize(image, layer).getMapId()["tile_fetcher"].url_format


def load_scene(asset):
    prefixes = (S2 + "/", L8 + "/", L9 + "/")
    if (
        not isinstance(asset, str)
        or not asset.startswith(prefixes)
        or not all(c.isalnum() or c in "/_-" for c in asset)
    ):
        raise ValueError("Unsupported scene identifier.")

    original = ee.Image(asset)
    if asset.startswith(S2 + "/"):
        image = ee.Image(
            ee.ImageCollection([original])
            .linkCollection(
                ee.ImageCollection(CLOUD_SCORE).filter(
                    ee.Filter.eq("system:index", asset[len(S2) + 1:])
                ),
                ["cs_cdf"]
            )
            .first()
        )
        return process_sentinel(image)
    if asset.startswith(L8 + "/"):
        return process_landsat(original, L8, "Landsat 8")
    return process_landsat(original, L9, "Landsat 9")


def compare_images(first, second, first_id, second_id, geom, area):
    delta = (
        ee.Image(second).select("NDVI")
        .subtract(ee.Image(first).select("NDVI"))
        .rename("CHANGE")
    )
    areas = ee.Image.cat([
        ee.Image.pixelArea().updateMask(delta.mask()).rename("valid"),
        ee.Image.pixelArea().updateMask(delta.lt(-.15)).rename("decline")
    ]).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom,
        scale=30, maxPixels=2000000, tileScale=4
    ).getInfo()
    valid = float(areas.get("valid") or 0)
    decline = float(areas.get("decline") or 0)
    average = delta.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom,
        scale=30, maxPixels=2000000, tileScale=4
    ).get("CHANGE").getInfo()

    return {
        "from": first_id, "to": second_id,
        "mean_delta": round(float(average), 5) if average is not None else None,
        "decline_pct": round(100 * decline / valid, 2) if valid else None,
        "paired_coverage_pct": coverage_pct(valid, area),
        "tile_url": tile_url(delta, "CHANGE", geom)
    }


@app.get("/")
def home():
    return FileResponse(
        ROOT / "static" / "index.html",
        headers={"Cache-Control": "no-store"}
    )


@app.get("/api/config")
def configuration():
    return {
        "mapbox_token": MAPBOX_TOKEN,
        "mapbox_style": MAPBOX_STYLE,
        "version": "geovibe-professional-v7"
    }


@app.get("/api/health")
def health():
    try:
        initialize_ee()
        ee.Number(1).getInfo()
        return {"status": "connected"}
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/analyze")
def analyze(request: AnalysisRequest):
    if (
        request.start < date(2017, 3, 28)
        or request.end > date.today()
        or request.start > request.end
    ):
        raise HTTPException(400, "Invalid study dates.")
    if (request.end - request.start).days > 4383:
        raise HTTPException(400, "Maximum study period: 12 years.")

    try:
        initialize_ee()
        geom = validate_geometry(request.geometry)
        mode, selected_ids = selected_periods(request.start, request.end)
        refs = reference_periods(mode, request.end, selected_ids)

        earliest_ref = date(
            int(refs[0][:4]),
            int(refs[0][5:7]) if len(refs[0]) == 7 else 5, 1
        )
        collection_start = min(request.start, earliest_ref)
        s, l8, l9 = get_collections(
            geom, collection_start, request.end
        )

        def in_requested_range(collection):
            return collection.filterDate(
                request.start.isoformat(),
                (request.end + timedelta(days=1)).isoformat()
            )

        selected_s = in_requested_range(s)
        selected_l8 = in_requested_range(l8)
        selected_l9 = in_requested_range(l9)
        counts = ee.Dictionary({
            "sentinel": selected_s.size(),
            "landsat8": selected_l8.size(),
            "landsat9": selected_l9.size()
        }).getInfo()

        total = sum(int(v) for v in counts.values())
        if total > MAX_SCENES:
            raise HTTPException(
                400, f"{total} scenes found. Limit: {MAX_SCENES}. "
                     "Shorten your period."
            )
        if not total:
            return {"observations": [], "message": "No scenes found."}

        area = float(geom.area(maxError=5).getInfo())
        landsat = l8.merge(l9)
        observations = scene_rows(
            selected_s.merge(selected_l8.merge(selected_l9)),
            total, geom, area
        )

        def valid(row):
            return all(row.get(key) is not None for key in INDICES)

        usable_scenes = [
            row for row in observations
            if row["coverage_pct"] >= SCENE_COVERAGE and valid(row)
        ]
        rgb_scenes = [
            row for row in observations
            if row["coverage_pct"] >= RGB_COVERAGE and valid(row)
        ]

        periods, images = calculate_periods(
            s, landsat, selected_ids, geom, area
        )
        reference, _ = calculate_periods(
            s, landsat, refs, geom, area
        )
        usable_periods = [
            row for row in periods
            if row["coverage_pct"] >= PERIOD_COVERAGE and valid(row)
        ]

        latest = (
            usable_periods[-1]
            if usable_periods else
            (usable_scenes[-1] if usable_scenes else None)
        )

        tiles = {}
        latest_image = (
            images[usable_periods[-1]["date"]]
            if usable_periods else
            (load_scene(usable_scenes[-1]["asset"]) if usable_scenes else None)
        )
        if latest_image is not None:
            for layer in ("NDVI", "NDMI", "EVI", "RGB"):
                tiles[layer] = tile_url(latest_image, layer, geom)

        change = None
        if len(usable_periods) >= 2:
            older, newer = usable_periods[-2], usable_periods[-1]
            change = compare_images(
                images[older["date"]], images[newer["date"]],
                older["date"], newer["date"], geom, area
            )
            tiles["CHANGE"] = change["tile_url"]

        rain, rain_error = None, None
        try:
            rain = precipitation_data(
                geom, request.start, request.end,
                selected_ids + refs
            )
        except Exception as exc:
            rain_error = str(exc)
            log.warning("NASA POWER error: %s", exc)

        return {
            "source": "Sentinel-2 + Landsat 8/9",
            "mode": mode,
            "observations": observations,
            "rgb_scenes": rgb_scenes,
            "periods": periods,
            "reference": reference,
            "score": historical_score(reference),
            "trends": {
                key: historical_trend(reference, key)
                for key in INDICES
            },
            "rainfall": rain,
            "rainfall_error": rain_error,
            "scene_counts": counts,
            "tiles": tiles,
            "change": change,
            "stats": {
                "area_ha": round(area / 10000, 2),
                "total_scenes": total,
                "usable_scenes": len(usable_scenes),
                "rgb_scenes": len(rgb_scenes),
                "usable_periods": len(usable_periods),
                "latest_usable": latest
            },
            "method_note": (
                "Sentinel-2 uses SCL and Cloud Score+. Landsat uses "
                "QA_PIXEL and QA_RADSAT masks. Indices share a "
                "common valid-pixel mask. Monthly precipitation "
                "is from NASA POWER. Historical indices compare "
                "matching calendar months or May-August seasons. "
                "The Crop Condition Score is an experimental "
                "historical rank, not a validated crop health score. "
                "Sentinel-2 and Landsat are not fully cross-calibrated."
            )
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception("Earth Engine analysis failed")
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/scene-layer")
def scene_layer(request: SceneRequest):
    if request.layer not in ("RGB", "NDVI", "NDMI", "EVI"):
        raise HTTPException(400, "Unsupported layer.")
    try:
        initialize_ee()
        geom = validate_geometry(request.geometry)
        image = load_scene(request.asset)
        return {
            "layer": request.layer,
            "tile_url": tile_url(image, request.layer, geom)
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception("Scene rendering failed")
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/scene-preview")
def scene_preview(request: SceneRequest):
    try:
        initialize_ee()
        geom = validate_geometry(request.geometry)
        image = load_scene(request.asset)
        return {
            "thumbnail_url": visualize(image, "RGB").getThumbURL({
                "region": geom.bounds(maxError=10),
                "dimensions": "240x135",
                "format": "png"
            })
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception("Thumbnail failed")
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/change-tile")
def change_tile(request: ChangeRequest):
    try:
        initialize_ee()
        geom = validate_geometry(request.geometry)
        a_start, _ = period_dates(request.period_from)
        _, b_end = period_dates(request.period_to)
        if request.period_from >= request.period_to:
            raise ValueError("Choose two ordered periods.")
        start = date.fromisoformat(a_start)
        end = date.fromisoformat(b_end) - timedelta(days=1)
        s, l8, l9 = get_collections(geom, start, end)
        landsat = l8.merge(l9)
        a = period_image(s, landsat, request.period_from)
        b = period_image(s, landsat, request.period_to)
        area = float(geom.area(maxError=5).getInfo())
        return compare_images(
            a, b, request.period_from, request.period_to, geom, area
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception("Change detection failed")
        raise HTTPException(502, str(exc)) from exc
