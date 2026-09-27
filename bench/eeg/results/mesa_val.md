# MESA-validatienachten (n = 80, NSRR-referentie, één scoorder per nacht)

| arm | n_pred/n_ref | gepoold sens | PPV | **F1** | mediaan F1/nacht | beter/slechter dan psgscoring | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|
| psgscoring 0.34.2 (n 80) | 11675/12432 | 0,546 | 0,581 | **0,563** | 0,556 | — | — |
| psgscoring 0.34.2 op dezelfde 80 nachten | 11675/12432 | 0,546 | 0,581 | **0,563** | 0,556 | (referentie voor de rij hieronder) | — |
| MSED τ 0,64 (upstream) (n 80) | 1672/12432 | 0,099 | 0,736 | **0,174** | 0,017 | 5/74 (Δmed -0,455) | 0,0000 |
| psgscoring 0.34.2 op dezelfde 79 nachten | 11459/12432 | 0,546 | 0,592 | **0,568** | 0,560 | (referentie voor de rij hieronder) | — |
| U-Net-50Hz (τ uit MESA-val) (n 79) | 11004/12432 | 0,643 | 0,726 | **0,682** | 0,673 | 74/5 (Δmed 0,112) | 0,0000 |

## MSED-drempelveeg op MESA-val (hier mag gekozen worden; PSG-IPA blijft schoon)

| τ | n_pred/n_ref | sens | PPV | gepoold F1 | mediaan F1/nacht |
|---:|---:|---:|---:|---:|---:|
| 0,30 | 6303/12432 | 0,188 | 0,372 | 0,250 | 0,127 |
| 0,40 | 4246/12432 | 0,161 | 0,473 | 0,241 | 0,094 |
| 0,50 | 2876/12432 | 0,134 | 0,580 | 0,218 | 0,047 |
| 0,55 | 2357/12432 | 0,121 | 0,639 | 0,204 | 0,028 |
| 0,60 | 1934/12432 | 0,107 | 0,690 | 0,186 | 0,018 |
| 0,64 | 1672/12432 | 0,099 | 0,736 | 0,174 | 0,017 |
| 0,70 | 1311/12432 | 0,084 | 0,793 | 0,151 | 0,014 |
| 0,75 | 1074/12432 | 0,072 | 0,836 | 0,133 | 0,003 |
| 0,80 | 865/12432 | 0,061 | 0,874 | 0,114 | 0,000 |
| 0,85 | 660/12432 | 0,048 | 0,897 | 0,090 | 0,000 |
| 0,90 | 441/12432 | 0,032 | 0,900 | 0,062 | 0,000 |

Beste MSED-drempel op MESA-val (gepoolde F1): **0,30**

## Per tertiel van arousallast (n_ref per nacht, gemeenschappelijke nachten)

| tertiel | n | n_ref-bereik | psgscoring F1 | U-Net F1 | ΔF1 |
|---|---:|---|---:|---:|---:|
| laag | 26 | 32–119 | 0,490 | 0,621 | 0,131 |
| midden | 27 | 126–172 | 0,553 | 0,672 | 0,119 |
| hoog | 26 | 185–370 | 0,611 | 0,712 | 0,102 |
