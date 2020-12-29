Based on https://github.com/nikhilweee/semantic-human-matting.

```
     ┌───────┐       
I ─> │ T-Net │ ─> trimap_pred <┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ L_t ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ trimap_gt <┄┐ 
│    └───────┘       │                                                           ┆
└─────> concat <─────┘                                                           ┆
          │                                                                      ┆
          ↓                                                                      ┆
      ┌───────┐                                                                  ┆
      | M-Net | ─> matte_u_pred <┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ L_m ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ matte_gt ┄┘
      └───────┘         

T-NET:   L_t Only
M-NET:   L_m Only, Use ground truth trimap instead of predicted trimap.
END2END: L_t:L_m = 1:100
```