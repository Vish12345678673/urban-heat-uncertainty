/*
 * YCEO-UHI
 * Google Earth Engine Predictor Extraction Workflow
 *
 * NOTE:
 * This is a documented reproducibility template based on the
 * predictor workflow used for the YCEO-UHI machine-learning dataset.
 *
 * The exact original interactive Earth Engine script was not recovered,
 * so source asset identifiers that cannot be verified are explicitly marked.
 */

// ================================================================
// 1. CONFIGURATION
// ================================================================

// Replace this with the exact YCEO urban-cluster FeatureCollection
// used in the original extraction.
var YCEO_CLUSTERS_ASSET = 'REPLACE_WITH_YOUR_YCEO_CLUSTER_ASSET';

var NUM_BATCHES = 21;


// ================================================================
// 2. YCEO UHI DATA
// ================================================================

var yceoUHI = ee.Image(
  'YALE/YCEO/UHI/UHI_all_averaged/v4'
)
.select('all_daytime_UHI')
.rename('Annual_day');

print('YCEO UHI:', yceoUHI);


// ================================================================
// 3. NDVI
// ================================================================

var ndvi = ee.ImageCollection(
  'MODIS/061/MOD13Q1'
)
.filterDate('2020-01-01', '2020-12-31')
.select('NDVI')
.mean()
.multiply(0.0001)
.rename('NDVI');

print('NDVI:', ndvi);


// ================================================================
// 4. ELEVATION
// ================================================================

var elevation = ee.Image(
  'USGS/SRTMGL1_003'
)
.select('elevation')
.rename('Elevation');

print('Elevation:', elevation);


// ================================================================
// 5. BUILT-UP SURFACE
// ================================================================

var builtup = ee.Image(
  'JRC/GHSL/P2023A/GHS_BUILT_S/2020'
)
.select(0)
.rename('Builtup');

print('Built-up:', builtup);


// ================================================================
// 6. POPULATION
// ================================================================

var population = ee.Image(
  'JRC/GHSL/P2023A/GHS_POP/2020'
)
.select(0)
.rename('Population');

print('Population:', population);


// ================================================================
// 7. CLIMATE PREDICTORS
// ================================================================
//
// The exact original climate source used in the interactive
// extraction was not recovered. ERA5-Land is therefore included
// here as a documented placeholder.
//
// Replace these assets with the exact source used for the final
// exported dataset if strict reproduction is required.
// ================================================================

var climate = ee.ImageCollection(
  'ECMWF/ERA5_LAND/MONTHLY_AGGR'
)
.filterDate('2020-01-01', '2020-12-31');

var meanTemp = climate
.select('temperature_2m')
.mean()
.subtract(273.15)
.rename('Mean_Temp_C');

var meanPrecip = climate
.select('total_precipitation_sum')
.mean()
.multiply(1000)
.rename('Mean_Monthly_Precip_mm');

print('Temperature:', meanTemp);
print('Precipitation:', meanPrecip);


// ================================================================
// 8. LAND COVER
// ================================================================

var worldCover = ee.ImageCollection(
  'ESA/WorldCover/v100'
)
.filter(ee.Filter.eq('MAP_YEAR', 2020))
.first()
.select('Map');


// Built-up land-cover fraction
var urbanMask = worldCover
.eq(50)
.rename('Urban_LC_Fraction');


// Water land-cover fraction
var waterMask = worldCover
.eq(80)
.rename('Water_LC_Fraction');


// Vegetation land-cover fraction
var vegetationMask = worldCover
.eq(10)
.or(worldCover.eq(20))
.or(worldCover.eq(30))
.or(worldCover.eq(90))
.or(worldCover.eq(95))
.or(worldCover.eq(100))
.rename('Vegetation_LC_Fraction');

print('Urban fraction:', urbanMask);
print('Water fraction:', waterMask);
print('Vegetation fraction:', vegetationMask);


// ================================================================
// 9. COMBINED PREDICTOR IMAGE
// ================================================================

var predictors = ndvi
.addBands(elevation)
.addBands(builtup)
.addBands(population)
.addBands(meanTemp)
.addBands(meanPrecip)
.addBands(urbanMask)
.addBands(waterMask)
.addBands(vegetationMask);

print('Predictor image:', predictors);
print('Number of bands:', predictors.bandNames().size());
print('Bands:', predictors.bandNames());


// ================================================================
// 10. LOAD YCEO CLUSTERS
// ================================================================

if (
  YCEO_CLUSTERS_ASSET ===
  'REPLACE_WITH_YOUR_YCEO_CLUSTER_ASSET'
) {

  print(
    'ERROR: Replace YCEO_CLUSTERS_ASSET with the actual ' +
    'YCEO cluster FeatureCollection asset.'
  );

} else {

  var clusters = ee.FeatureCollection(
    YCEO_CLUSTERS_ASSET
  );

  print(
    'YCEO clusters:',
    clusters.size()
  );


  // ==============================================================
  // 11. CREATE 21 EXPORT BATCHES
  // ==============================================================

  var totalClusters = clusters.size();

  var batchSize = totalClusters
    .divide(NUM_BATCHES)
    .ceil();

  print(
    'Total clusters:',
    totalClusters
  );

  print(
    'Number of batches:',
    NUM_BATCHES
  );

  print(
    'Approximate batch size:',
    batchSize
  );


  // ==============================================================
  // 12. EXPORT CLUSTER-LEVEL PREDICTORS
  // ==============================================================

  for (var batch = 0; batch < NUM_BATCHES; batch++) {

    var start = batchSize.multiply(batch);

    var batchClusters = ee.FeatureCollection(
      clusters.toList(
        batchSize,
        start
      )
    );


    var extracted = predictors.reduceRegions({
      collection: batchClusters,
      reducer: ee.Reducer.mean(),
      scale: 1000,
      tileScale: 4
    });


    Export.table.toDrive({

      collection: extracted,

      description:
        'YCEO_Enhanced_Batch_' + batch,

      fileNamePrefix:
        'YCEO_Enhanced_Batch_' + batch,

      fileFormat: 'CSV'
    });

  }


  print(
    'Export tasks created:',
    NUM_BATCHES
  );

}


// ================================================================
// 13. MAP VISUALIZATION
// ================================================================

Map.addLayer(
  yceoUHI,
  {
    min: -1.5,
    max: 7.5
  },
  'YCEO Annual Daytime UHI'
);

Map.addLayer(
  ndvi,
  {
    min: 0,
    max: 1
  },
  'NDVI'
);

Map.addLayer(
  elevation,
  {},
  'Elevation'
);

Map.addLayer(
  builtup,
  {},
  'Built-up Surface'
);

Map.addLayer(
  population,
  {},
  'Population'
);

print(
  'Final predictor bands:',
  predictors.bandNames()
);