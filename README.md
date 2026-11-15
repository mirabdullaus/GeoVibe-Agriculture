# GeoVibe Agriculture — clean start v0.1

A local FastAPI app with Mapbox drawing, a selected example study area and Google Earth Engine Sentinel-2 image layers (RGB, NDVI, NDMI, NDRE, EVI). It computes per-scene index means for a polygon without downloading raster files.

## Windows PowerShell instructions

1. Open this folder in VS Code, open Terminal.
2. Create Python environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. Configure `.env` by copying `.env.example` and editing it:
   ```powershell
   Copy-Item .env.example .env
   ```
   Set `MAPBOX_TOKEN=pk...` to your own public Mapbox token. Set EE_PROJECT to the actual Earth Engine Google Cloud project ID (sample is prefilled). Keep `.env` private.
4. Authenticate (once if not already done for your user account):
   ```powershell
   .\.venv\Scripts\earthengine.exe authenticate
   ```
5. Run:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
   ```
6. Open http://127.0.0.1:8000

## Notes / limitations

- Example polygon is a *sample study area*, **not** a verified farm boundary. You may draw a different polygon.
- Sentinel-2 Level-2A surface reflectance is used; the scene classification layer masks problematic cloud/shadow/snow pixels. The screening is useful for a demo, not a full validation or cloud-probability model.
- The defaults analyze May–August 2024. Limit is 125 days / 55 scenes and approx. 0.12° bounding box. Bigger and longer requests need asynchronous jobs and cost controls.
- Means are based on individual satellite acquisitions and are not daily measurements, seasonal trends or crop-health diagnoses. The time series excludes scenes where less than 20% of the field has accepted pixels (an exploratory threshold).
- `NDMI` is not direct soil moisture; `NDRE` is not direct chlorophyll content. Sentinel-2 cannot retrieve land surface temperature directly.
- The map tile uses the most recent acquisition within the period. Clouds/shadows remain transparent; a latest scene with low coverage may appear mostly blank. The numeric snapshot is the most recent scene with ≥20% good coverage.
- These are real server requests and may incur Earth Engine resource usage. An authenticated Google account with a registered project is required.
- This prototype is for local use only; add login, rate limiting, caching, job queue, stable tiles, privacy protections and server-side credentials before public deployment.

Data: `COPERNICUS/S2_SR_HARMONIZED` (ESA/Copernicus, via Google Earth Engine). Map by Mapbox.
