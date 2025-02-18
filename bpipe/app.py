from hpo_threshpoint import optimize_adj_value

mod_fn = "/media/ljp238/12TBWolf/RSPROX/OUT_TILES/BLOCKS/TLS/tiles/N13E103_DTM_GV4.tif"
ref_fn = "/media/ljp238/12TBWolf/RSPROX/OUT_TILES/TILES12/N13E103/N13E103_cdem_DEM.tif"


timeout=900 
maximize = optimize_adj_value(mod_fn, ref_fn,
                              lwm_fpath=None, gridshape=(9001,9001),
                              direction="maximize", n_trials=500, timeout=1000)

minimize = optimize_adj_value(mod_fn, ref_fn,
                              lwm_fpath=None, gridshape=(9001,9001),
                              direction="minimize", n_trials=500, timeout=1000)