Based on https://github.com/nikhilweee/semantic-human-matting.

```
     ┌───────┐       ┌┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ L_t ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐
I ─> │ T-Net │ ─> trimap_pred ───┐                                   trimap_gt <┄┐ 
│    └───────┘       │           │                                               ┆
└─────> concat <─────┘           │             ┌───────┐                         ┆
└─────────┼──────────────────────┼─> concat ─> │ F-Net │ ─> matte_pred <┄L_f┄┄ matte_gt
          ↓                      │             └───────┘                         ┆
      ┌───────┐                  │                                               ┆
      | M-Net | ─> matte_u_pred ─┘                                  alpha_u_gt <┄┘
      └───────┘         └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ L_m ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘
                     (the same unkown area between matte_u and matte_u_pred)
                     (when pretrain M-Net, use trimap_gt's area, otherwise trimap_pred)


T-NET:   L_t Only
M-NET:   L_m Only, Use ground truth trimap instead of predicted trimap.
END2END: L_t:L_m = 1:100

HR train step:
1. T-Net(LR)
2. M-Net(LR)
3. F-Net(LR)
4. end2end(LR)
5. M-Net(HR)
6. F-Net(HR)
HR test step:
image(HR) -> image(LR) -> T-Net -> trimap(LR) -> trimap(HR)
    |                                              |
    ------------------------+-----------------------
                            |
                      M-Net & F-Net
                            |
                         matte(HR)
```