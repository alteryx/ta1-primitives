cd datasets
git lfs pull -I datasets/seed_datasets_current/32_wikiqa/
git lfs pull -I seed_datasets_current/196_autoMpg/
git lfs pull -I seed_datasets_current/1491_one_hundred_plants_margin/
git lfs pull -I seed_datasets_current/kaggle_music_hackathon/
git lfs pull -I seed_datasets_current/LL0_1100_popularkids/
git lfs pull -I seed_datasets_current/LL0_acled_reduced/
git lfs pull -I seed_datasets_current/LL1_50words/
git lfs pull -I seed_datasets_current/LL1_736_population_spawn_simpler/
git lfs pull -I seed_datasets_current/LL1_736_population_spawn/
git lfs pull -I seed_datasets_current/LL1_736_stock_market/
git lfs pull -I training_datasets/LL1/LL1_retail_sales_binary/
git lfs pull -I training_datasets/LL1/LL1_retail_sales_multi/
git lfs pull -I seed_datasets_current/LL1_retail_sales_total/
git lfs pull -I seed_datasets_current/loan_status/
# The following datasets are not needed for the test pipeplines - only to prevent error messages during pipeline runs
git lfs pull -I training_datasets/LL0/LL0_1038_gina_agnostic/
git lfs pull -I training_datasets/LL0/LL0_300_isolet/
git lfs pull -I training_datasets/LL0/LL0_1515_micro_mass/
git lfs pull -I training_datasets/LL0/LL0_1041_gina_prior2/
git lfs pull -I training_datasets/LL0/LL0_1176_internet_advertisements/
git lfs pull -I training_datasets/LL0/LL0_4134_bioresponse/
git lfs pull -I training_datasets/LL0/LL0_1457_amazon_commerce_reviews/
git lfs pull -I training_datasets/LL0/LL0_1122_ap_breast_prostate/
git lfs pull -I training_datasets/LL0/LL0_1468_cnae_9/
git lfs pull -I seed_datasets_current/LL1_VTXC_1343_cora/
git lfs pull -I training_datasets/LL1/LL1_1233_eating/
cd ..