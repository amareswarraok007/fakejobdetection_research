# e5_explain

Source: `e5_explain/20261120-062154` · git `2cb11a5e0b6cca94dd2fba7315d5e7ad611ef492`

## SHAP correctness checks

| Model | Kind | Check | Value | Limit | Passed |
| --- | --- | --- | --- | --- | --- |
| lr_text | linear | max |formula - LinearExplainer| | 0.0000000000 | 0.000001 | True |
| lr_full | linear | max |formula - LinearExplainer| | 0.0000000000 | 0.000001 | True |

## Global top features: lr_text

| Rank | Feature | Mean |SHAP| |
| --- | --- | --- |
| 1 | `text__company description` | 0.1928 |
| 2 | `text__web` | 0.0720 |
| 3 | `text__urltoken` | 0.0643 |
| 4 | `text__it` | 0.0504 |
| 5 | `text__data` | 0.0490 |
| 6 | `text__our` | 0.0469 |
| 7 | `text__software` | 0.0439 |
| 8 | `text__customer service` | 0.0416 |
| 9 | `text__customer` | 0.0412 |
| 10 | `text__digital` | 0.0397 |
| 11 | `text__english` | 0.0396 |
| 12 | `text__service` | 0.0370 |
| 13 | `text__companies` | 0.0366 |
| 14 | `text__marketing` | 0.0359 |
| 15 | `text__your` | 0.0347 |
| 16 | `text__apply` | 0.0346 |
| 17 | `text__no` | 0.0344 |
| 18 | `text__full` | 0.0330 |
| 19 | `text__growing` | 0.0329 |
| 20 | `text__clients` | 0.0329 |
| 21 | `text__office` | 0.0326 |
| 22 | `text__knowledge of` | 0.0324 |
| 23 | `text__assistant` | 0.0316 |
| 24 | `text__industry` | 0.0309 |
| 25 | `text__offer` | 0.0306 |
| 26 | `text__recruitment` | 0.0301 |
| 27 | `text__new` | 0.0299 |
| 28 | `text__engineering` | 0.0291 |
| 29 | `text__client` | 0.0286 |
| 30 | `text__mobile` | 0.0276 |

## Global top features: lr_full

| Rank | Feature | Mean |SHAP| |
| --- | --- | --- |
| 1 | `numeric__has_company_logo` | 0.5276 |
| 2 | `numeric__company_profile_empty` | 0.5062 |
| 3 | `numeric__log_text_length` | 0.3959 |
| 4 | `category__required_experience_Associate` | 0.3039 |
| 5 | `category__required_education_Bachelor's Degree` | 0.2802 |
| 6 | `category__required_experience_Unknown` | 0.2411 |
| 7 | `category__employment_type_Full-time` | 0.2259 |
| 8 | `category__required_education_High School or equivalent` | 0.2240 |
| 9 | `function__function_Engineering` | 0.2089 |
| 10 | `numeric__n_url` | 0.1972 |
| 11 | `numeric__n_email` | 0.1891 |
| 12 | `function__function_Sales` | 0.1689 |
| 13 | `numeric__requirements_empty` | 0.1683 |
| 14 | `category__required_education_Unknown` | 0.1663 |
| 15 | `category__required_experience_Mid-Senior level` | 0.1624 |
| 16 | `function__function_Information Technology` | 0.1301 |
| 17 | `numeric__uppercase_ratio` | 0.1195 |
| 18 | `function__function_Administrative` | 0.1181 |
| 19 | `numeric__money_mentions` | 0.1160 |
| 20 | `numeric__benefits_empty` | 0.1059 |
| 21 | `function__function_Customer Service` | 0.0992 |
| 22 | `numeric__salary_given` | 0.0983 |
| 23 | `category__employment_type_Temporary` | 0.0939 |
| 24 | `function__function_Health Care Provider` | 0.0922 |
| 25 | `function__function_Education` | 0.0834 |
| 26 | `category__required_experience_Entry level` | 0.0810 |
| 27 | `numeric__exclamation_count` | 0.0715 |
| 28 | `category__employment_type_Contract` | 0.0695 |
| 29 | `category__employment_type_Unknown` | 0.0642 |
| 30 | `text__web` | 0.0620 |
