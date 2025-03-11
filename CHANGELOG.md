## v0.4.1 (2025-03-11)

### Fix

- add support for SciPy v1.15 (#39)
- upgrade scaffolding and support sklearn v1.6 (#34)

## v0.4.0 (2024-06-11)

### Feat

- improve sample efficiency (#28)
- simplify installation (#26)
- support pre-fitted estimators (#19)

## v0.3.1 (2024-04-19)

### Fix

- fix imports when darts is not available (#16)

## v0.3.0 (2024-04-14)

### Feat

- add Darts forecaster (#14)

## v0.2.4 (2024-04-01)

### Fix

- only convert dtype when target dtype is not integer (#12)

## v0.2.3 (2024-04-01)

### Fix

- improve dtype handling in CoherentLinearQuantileRegressor (#10)

## v0.2.2 (2024-04-01)

### Fix

- xfail sklearn's check_regressors_int (#9)

## v0.2.1 (2024-03-30)

### Fix

- round quantiles when target is integer (#7)

## v0.2.0 (2024-03-30)

### Feat

- improve coherence with intercept clipping (#5)

### Fix

- round when target is integer (#6)

## v0.1.1 (2024-03-17)

### Build

- make dependency on numpy and scipy explicit (#4)

## v0.1.0 (2024-03-16)

### Feat

- add Conformal Coherent Quantile Regression (#3)
