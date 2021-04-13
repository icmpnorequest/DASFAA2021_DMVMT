# Raw Data Description


## 1. Raw Data: TrustPilot User Reviews

In the experiments, we use five real-world user reviews dataset proposed by Hovy et al. [1]: 

- denmark.auto-adjusted_gender.NUTS-regions.jsonl.tmp
- germany.auto-adjusted_gender.NUTS-regions.jsonl.tmp
- germany.auto-adjusted_gender.NUTS-regions.jsonl.tmp
- united_kingdom.auto-adjusted_gender.NUTS-regions.jsonl.tmp
- united_states.auto-adjusted_gender.geocoded.jsonl.tmp

The raw data is too large to upload on Github, so you could download the raw datasets from Hovy's provided [link](https://bitbucket.org/lowlands/release/src/master/WWW2015/data/).

After downloading all these five files, we need to copy these five files under the `WWW2015_raw` directory, listing as follow:

```txt
WWW2015_raw
├── denmark.auto-adjusted_gender.NUTS-regions.jsonl.tmp
├── france.auto-adjusted_gender.NUTS-regions.jsonl.tmp
├── germany.auto-adjusted_gender.NUTS-regions.jsonl.tmp
├── united_kingdom.auto-adjusted_gender.NUTS-regions.jsonl.tmp
└── united_states.auto-adjusted_gender.geocoded.jsonl.tmp

0 directories, 5 files
```



## 2. Citations

[1] Hovy, D., Johannsen, A., Søgaard, A.: User review sites as a resource for large- scale sociolinguistic studies. In: Proceedings of the 24th international conference on World Wide Web. pp. 452–461 (2015)