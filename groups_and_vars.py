from used_packages import *

def group_names_and_variables():

    group_names = {
        0: "Geographic ID",
        1: "CA Name",
        2: "Total Pop. 2000",
        3: "Total Pop. 2010",
        4: "Total Pop. 2020",
        5: "Total Households 2020",
        6: "Avg. Household Size 2020",
        7: "Total Pop. 2022",
        8: "Age Cohort",
        9: "Race and Ethnicity",
        10: "Pop. living in households",
        11: "Pop. aged 16 and over",
        12: "Employment Status",
        13: "Mode of Travel to Work",
        14: "Aggregate travel time to work",
        15: "Vehicles Available",
        16: "Pop. aged 25 and over",
        17: "Education",
        18: "Household Income",
        19: "Household Occupancy",
        20: "Housing Type",
        21: "Housing Size",
        22: "Housing Age",
        23: "Home Value",
        24: "Rent",
        25: "Household PC & Net Access",
        26: "Disability No.",
        27: "Disability by Type",
        28: "Disability by Age",
        29: "Avg. Vehicles Miles",
        30: "Sales",
        31: "Equalized Assessed Value",
        32: "General Land Use",
        33: "Household Size",
        34: "Household Type",
        35: "Nativity",
        36: "Language"
    }

    groups_dict = {
        "GEOID": 0,
        "GEOG": 1,
        "2000_POP": 2,
        "2010_POP": 3,
        "2020_POP": 4,
        "2020_HH": 5,
        "2020_HH_SIZE": 6,
        "TOT_POP": 7,
        "UND5": 8, "A5_19": 8, "A20_34": 8, "A35_49": 8, "A50_64": 8, "A65_74": 8, "A75_84": 8, "OV85": 8, "MED_AGE": 8,
        "WHITE": 9, "HISP": 9, "BLACK": 9, "ASIAN": 9, "OTHER": 9,
        "POP_HH": 10,
        "POP_16OV": 11,
        "IN_LBFRC": 12, "EMP": 12, "UNEMP": 12, "NOT_IN_LBFRC": 12,
        "TOT_WRKR16OV": 13, "WORK_AT_HOME": 13, "TOT_COMM": 13, "DROVE_AL": 13, "CARPOOL": 13, "TRANSIT": 13, "WALK_BIKE": 13, "COMM_OTHER": 13,
        "AGG_TT": 14,
        "NO_VEH": 15, "ONE_VEH": 15, "TWO_VEH": 15, "THREEOM_VEH": 15,
        "POP_25OV": 16,
        "LT_HS": 17, "HS": 17, "SOME_COLL": 17, "ASSOC": 17, "BACH": 17, "GRAD_PROF": 17,
        "INC_LT_25K": 18, "INC_25_50K": 18, "INC_50_75K": 18, "INC_75_100K": 18, "INC_100_150K": 18, "INC_GT_150": 18, "MEDINC": 18, "INCPERCAP": 18,
        "TOT_HH": 19, "OWN_OCC_HU": 19, "RENT_OCC_HU": 19, "VAC_HU": 19,
        "HU_TOT": 20, "HU_SNG_DET": 20, "HU_SNG_ATT": 20, "HU_2UN": 20, "HU_3_4UN": 20, "HU_5_9UN": 20, "HU_10_19UN": 20, "HU_GT_19UN": 20, "HU_MOBILE": 20,
        "MED_ROOMS": 21, "BR_0_1": 21, "BR_2": 21, "BR_3": 21, "BR_4": 21, "BR_5": 21,
        "HA_AFT2010": 22, "HA_90_10": 22, "HA_70_90": 22, "HA_40_70": 22, "HA_BEF1940": 22, "MED_HA": 22,
        "HV_LT_150K": 23, "HV_150_300K": 23, "HV_300_500K": 23, "HV_GT_500K": 23, "MED_HV": 23,
        "CASHRENT_HH": 24, "RENT_LT500": 24, "RENT_500_999": 24, "RENT_1000_1499": 24, "RENT_1500_2499": 24, "RENT_GT2500": 24, "MED_RENT": 24,
        "COMPUTER": 25, "ONLY_SMARTPHONE": 25, "NO_COMPUTER": 25, "INTERNET": 25, "BROADBAND": 25, "NO_INTERNET": 25,
        "DISAB_ONE": 26, "DISAB_TWOMOR": 26, "DISAB_ANY": 26,
        "DIS_HEAR": 27, "DIS_VIS": 27, "DIS_COG": 27, "DIS_AMB": 27, "DIS_SLFCARE": 27, "DIS_INDPLIV": 27,
        "DIS_UND18": 28, "DIS_18_64": 28, "DIS_65_75": 28, "DIS_75OV": 28,
        "AVG_VMT": 29,
        "RET_SALES": 30, "GEN_MERCH": 30,
        "RES_EAV": 31, "CMRCL_EAV": 31, "IND_EAV": 31, "RAIL_EAV": 31, "FARM_EAV": 31, "MIN_EAV": 31, "TOT_EAV": 31,
        "TOT_ACRES": 32, "SF": 32, "Sfperc": 32, "MF": 32, "Mfperc": 32, "MIX": 32, "MIXperc": 32, "COMM": 32, "COMMperc": 32,
        "INST": 32, "INSTperc": 32, "IND": 32, "INDperc": 32, "TRANS": 32, "TRANSperc": 32, "AG": 32, "Agperc": 32, "OPEN": 32, "OPENperc": 32,
        "VACANT": 32, "VACperc": 32,
        "CT_1PHH": 33, "CT_2PHH": 33, "CT_3PHH": 33, "CT_4MPHH": 33,
        "CT_FAM_HH": 34, "CT_SP_WCHILD": 34, "CT_NONFAM_HH": 34,
        "NATIVE": 35, "FOR_BORN": 35,
        "NOT_ENGLISH": 36, "LING_ISO": 36, "ENGLISH": 36, "SPANISH": 36, "SLAVIC": 36, "CHINESE": 36, "TAGALOG": 36, "ARABIC": 36, "KOREAN": 36, "OTHER_ASIAN": 36, "OTHER_EURO": 36, "OTHER_UNSPEC": 36,
    }

    features_keep = [
         'Crime_Count',
         'CA', # cat
         'GEOG', # cat
         '2020_POP',
         '2020_HH', # total houholds
         '2020_HH_SIZE', # avg HH size
         'TOT_POP', # 2018-2022
         'UND5',
         'A5_19',
         'A20_34',
         'A35_49',
         'A50_64',
         'A65_74',
         'A75_84',
         'OV85',
         #'MED_AGE',
         'EMP',
         'UNEMP',
         'TOT_WRKR16OV',
         'WORK_AT_HOME',
         'TOT_COMM',
         'DROVE_AL',
         'CARPOOL',
         'TRANSIT',
         'WALK_BIKE',
         'COMM_OTHER',
         'AGG_TT', # aggregate travel time to work
         'NO_VEH',
         'ONE_VEH',
         'TWO_VEH',
         'THREEOM_VEH',
         'LT_HS', # less than high school
         'HS', # high school
         'SOME_COLL',
         'ASSOC',
         'BACH',
         'GRAD_PROF',
         'INC_LT_25K',
         'INC_25_50K',
         'INC_50_75K',
         'INC_75_100K',
         'INC_100_150K',
         'INC_GT_150',
         #'MEDINC',
         'INCPERCAP',
         'TOT_HH',
         'OWN_OCC_HU',
         'RENT_OCC_HU',
         'VAC_HU',
         #'HU_TOT', # total housing units
         'HU_SNG_DET',
         'HU_SNG_ATT',
         'HU_2UN',
         'HU_3_4UN',
         'HU_5_9UN',
         'HU_10_19UN',
         'HU_GT_19UN',
         'HU_MOBILE',
         #'MED_ROOMS',
         'BR_0_1', # bedrooms 0 to 1
         'BR_2',
         'BR_3',
         'BR_4',
         'BR_5',
         'HA_AFT2010',
         'HA_90_10',
         'HA_70_90',
         'HA_40_70',
         'HA_BEF1940',
         #'MED_HA',
         'HV_LT_150K', # home value
         'HV_150_300K',
         'HV_300_500K',
         'HV_GT_500K',
         #'MED_HV',
         'CASHRENT_HH',
         'RENT_LT500',
         'RENT_500_999',
         'RENT_1000_1499',
         'RENT_1500_2499',
         'RENT_GT2500',
         #'MED_RENT',
         'COMPUTER',
         'ONLY_SMARTPHONE',
         'NO_COMPUTER',
         'INTERNET',
         'BROADBAND',
         'NO_INTERNET',
         'DISAB_ONE',
         'DISAB_TWOMOR',
         'DISAB_ANY', # total no of people with disability
         'DIS_HEAR',
         'DIS_VIS',
         'DIS_COG',
         'DIS_AMB',
         'DIS_SLFCARE',
         'DIS_INDPLIV',
         'DIS_UND18',
         'DIS_18_64',
         'DIS_65_75',
         'DIS_75OV',
         'AVG_VMT', # average vehicle miles traveled
         'TOT_ACRES',
         #'SF',
         'Sfperc',
         #'MF',
         'Mfperc',
         #'MIX',
         'MIXperc',
         #'COMM',
         'COMMperc',
         #'INST',
         'INSTperc',
         #'IND',
         'INDperc',
         #'TRANS',
         'TRANSperc',
         #'AG',
         'Agperc',
         #'OPEN',
         'OPENperc',
         #'VACANT',
         'VACperc',
         'CT_1PHH', # 1 person HH
         'CT_2PHH',
         'CT_3PHH',
         'CT_4MPHH',
         'CT_FAM_HH', # family
         'CT_SP_WCHILD', # single parent with child
         'CT_NONFAM_HH',
         'NATIVE',
         'FOR_BORN',
         'NOT_ENGLISH',
         'LING_ISO',
         'ENGLISH',
         'SPANISH',
         'SLAVIC',
         'CHINESE',
         'TAGALOG',
         'ARABIC',
         'KOREAN',
         'OTHER_ASIAN',
         'OTHER_EURO',
         'OTHER_UNSPEC',
         ]
    return features_keep, groups_dict, group_names

def create_distilled_features(df):
    """
    Create distilled features from demographic data groups.

    Parameters:
    df (pandas.DataFrame): DataFrame containing all the original demographic features

    Returns:
    pandas.DataFrame: A DataFrame with distilled features
    """
    # Create a new DataFrame for distilled features
    distilled_features = pd.DataFrame(index=df.index)

    # Population (groups 2-7)
    distilled_features['recent_population'] = df['TOT_POP'] if 'TOT_POP' in df.columns else df['2020_POP']
    distilled_features['avg_household_size'] = df['2020_HH_SIZE']

    # Age Cohort (8)
    distilled_features['median_age'] = df['MED_AGE']
    distilled_features['age_dependency'] = (df['UND5'] + df['OV85']) / (df['A20_34'] + df['A35_49'] + df['A50_64'])
    distilled_features['youth_ratio'] = df['UND5'] / distilled_features['recent_population']
    distilled_features['senior_ratio'] = (df['A65_74'] + df['A75_84'] + df['OV85']) / distilled_features['recent_population']

    # Race and Ethnicity (9)
    race_cols = ['WHITE', 'HISP', 'BLACK', 'ASIAN', 'OTHER']
    race_proportions = df[race_cols].div(df[race_cols].sum(axis=1), axis=0)
    # Diversity index (1 - sum of squared proportions) - higher means more diverse
    distilled_features['diversity_index'] = 1 - (race_proportions ** 2).sum(axis=1)
    # Largest demographic group percentage
    distilled_features['largest_demo_pct'] = df[race_cols].max(axis=1) / df[race_cols].sum(axis=1)

    # Population in households (10)
    distilled_features['hh_population_ratio'] = df['POP_HH'] / distilled_features['recent_population']

    # Employment Status (12)
    distilled_features['employment_rate'] = df['EMP'] / df['POP_16OV']
    distilled_features['unemployment_rate'] = df['UNEMP'] / df['IN_LBFRC']
    distilled_features['labor_participation'] = df['IN_LBFRC'] / df['POP_16OV']

    # Travel to Work (13-14)
    distilled_features['transit_use_rate'] = df['TRANSIT'] / df['TOT_COMM']
    distilled_features['work_from_home_rate'] = df['WORK_AT_HOME'] / df['TOT_WRKR16OV']
    distilled_features['car_commute_rate'] = (df['DROVE_AL'] + df['CARPOOL']) / df['TOT_COMM']
    distilled_features['active_commute_rate'] = df['WALK_BIKE'] / df['TOT_COMM']
    # Average travel time per commuter
    distilled_features['avg_commute_time'] = df['AGG_TT'] / df['TOT_COMM']

    # Vehicles Available (15)
    distilled_features['car_ownership_rate'] = (df['ONE_VEH'] + df['TWO_VEH'] + df['THREEOM_VEH']) / df['TOT_HH']
    distilled_features['multi_car_rate'] = (df['TWO_VEH'] + df['THREEOM_VEH']) / df['TOT_HH']
    distilled_features['zero_car_rate'] = df['NO_VEH'] / df['TOT_HH']

    # Education (17)
    distilled_features['higher_education_rate'] = (df['BACH'] + df['GRAD_PROF']) / df['POP_25OV']
    distilled_features['hs_completion_rate'] = (df['HS'] + df['SOME_COLL'] + df['ASSOC'] + df['BACH'] + df['GRAD_PROF']) / df['POP_25OV']
    distilled_features['college_exposure_rate'] = (df['SOME_COLL'] + df['ASSOC'] + df['BACH'] + df['GRAD_PROF']) / df['POP_25OV']

    # Income (18)
    distilled_features['median_income'] = df['MEDINC']
    distilled_features['income_per_capita'] = df['INCPERCAP']
    distilled_features['high_income_pct'] = (df['INC_100_150K'] + df['INC_GT_150']) / df['TOT_HH']
    distilled_features['low_income_pct'] = df['INC_LT_25K'] / df['TOT_HH']

    # Housing Occupancy (19)
    distilled_features['homeownership_rate'] = df['OWN_OCC_HU'] / df['TOT_HH']
    distilled_features['rental_rate'] = df['RENT_OCC_HU'] / df['TOT_HH']
    distilled_features['vacancy_rate'] = df['VAC_HU'] / df['HU_TOT']

    # Housing Type (20)
    distilled_features['single_family_pct'] = (df['HU_SNG_DET'] + df['HU_SNG_ATT']) / df['HU_TOT']
    distilled_features['small_multifamily_pct'] = (df['HU_2UN'] + df['HU_3_4UN'] + df['HU_5_9UN']) / df['HU_TOT']
    distilled_features['large_multifamily_pct'] = (df['HU_10_19UN'] + df['HU_GT_19UN']) / df['HU_TOT']

    # Housing Size (21)
    distilled_features['median_rooms'] = df['MED_ROOMS']
    distilled_features['large_homes_pct'] = (df['BR_4'] + df['BR_5']) / (df['BR_0_1'] + df['BR_2'] + df['BR_3'] + df['BR_4'] + df['BR_5'])
    distilled_features['small_homes_pct'] = df['BR_0_1'] / (df['BR_0_1'] + df['BR_2'] + df['BR_3'] + df['BR_4'] + df['BR_5'])

    # Housing Age (22)
    distilled_features['median_home_age'] = df['MED_HA']
    distilled_features['new_housing_pct'] = (df['HA_AFT2010'] + df['HA_90_10']) / df['HU_TOT']
    distilled_features['old_housing_pct'] = df['HA_BEF1940'] / df['HU_TOT']

    # Home Value (23)
    distilled_features['median_home_value'] = df['MED_HV']
    distilled_features['high_value_homes_pct'] = (df['HV_300_500K'] + df['HV_GT_500K']) / (df['HU_TOT'] - df['VAC_HU'])

    # Rent (24)
    distilled_features['median_rent'] = df['MED_RENT']
    distilled_features['high_rent_pct'] = (df['RENT_1500_2499'] + df['RENT_GT2500']) / df['CASHRENT_HH']
    distilled_features['rent_to_income'] = (df['MED_RENT'] * 12) / df['MEDINC']

    # Computer & Internet Access (25)
    distilled_features['internet_access_rate'] = df['INTERNET'] / df['TOT_HH']
    distilled_features['broadband_rate'] = df['BROADBAND'] / df['TOT_HH']
    distilled_features['digital_divide_rate'] = df['NO_COMPUTER'] / df['TOT_HH']

    # Disability (26-28)
    distilled_features['disability_rate'] = df['DISAB_ANY'] / distilled_features['recent_population']
    distilled_features['youth_disability_rate'] = df['DIS_UND18'] / (df['UND5'] + df['A5_19'])
    distilled_features['senior_disability_rate'] = (df['DIS_65_75'] + df['DIS_75OV']) / (df['A65_74'] + df['A75_84'] + df['OV85'])

    # Vehicle Miles (29)
    if 'AVG_VMT' in df.columns:
        distilled_features['avg_vehicle_miles'] = df['AVG_VMT']


    # Land Use (32)
    distilled_features['residential_land_pct'] = df['Sfperc'] + df['Mfperc']
    distilled_features['commercial_industrial_pct'] = df['COMMperc'] + df['INDperc']
    distilled_features['open_space_pct'] = df['OPENperc']
    distilled_features['vacant_land_pct'] = df['VACperc']

    # Household Size (33)
    distilled_features['single_person_hh_rate'] = df['CT_1PHH'] / df['TOT_HH']
    distilled_features['large_hh_rate'] = df['CT_4MPHH'] / df['TOT_HH']

    # Household Type (34)
    distilled_features['family_hh_rate'] = df['CT_FAM_HH'] / df['TOT_HH']
    distilled_features['single_parent_rate'] = df['CT_SP_WCHILD'] / df['TOT_HH']

    # Nativity (35)
    distilled_features['foreign_born_pct'] = df['FOR_BORN'] / distilled_features['recent_population']

    # Language (36)
    distilled_features['non_english_pct'] = df['NOT_ENGLISH'] / distilled_features['recent_population']
    distilled_features['linguistic_isolation_rate'] = df['LING_ISO'] / df['TOT_HH']

    # Cross-group relationships
    distilled_features['home_value_to_income'] = df['MED_HV'] / df['MEDINC']
    distilled_features['population_density'] = distilled_features['recent_population'] / df['TOT_ACRES']

    distilled_features['pop_growth_rate_10yr'] = (df['2020_POP'] - df['2010_POP']) / df['2010_POP'] if '2010_POP' in df.columns else np.nan
    distilled_features['pop_growth_rate_20yr'] = (df['2020_POP'] - df['2000_POP']) / df['2000_POP'] if '2000_POP' in df.columns else np.nan
    distilled_features['recent_growth_rate'] = (df['TOT_POP'] - df['2020_POP']) / df['2020_POP'] if 'TOT_POP' in df.columns and '2020_POP' in df.columns else np.nan

    # Housing affordability metrics
    distilled_features['years_to_buy_median_home'] = df['MED_HV'] / df['INCPERCAP']
    distilled_features['rent_burden'] = (df['MED_RENT'] * 12) / (df['MEDINC'] / 3)  # Rent as percentage of 1/3 of annual income

    # Age distribution metrics
    distilled_features['child_ratio'] = (df['UND5'] + df['A5_19']) / distilled_features['recent_population']
    distilled_features['working_age_ratio'] = (df['A20_34'] + df['A35_49'] + df['A50_64']) / distilled_features['recent_population']
    distilled_features['millennial_zoomer_ratio'] = df['A20_34'] / (df['A35_49'] + df['A50_64'])

    # Mixed use and urban form metrics
    distilled_features['mixed_use_ratio'] = df['MIXperc'] / (df['Sfperc'] + df['Mfperc'] + df['COMMperc'] + df['INDperc'] + 0.001)  # Adding small constant to avoid division by zero
    distilled_features['walkability_proxy'] = (distilled_features['residential_land_pct'] * distilled_features['population_density'] * distilled_features['commercial_industrial_pct'] * distilled_features['active_commute_rate']) ** 0.25  # Geometric mean of factors

    # Social vulnerability metrics
    distilled_features['social_vulnerability_index'] = (
                                                               distilled_features['disability_rate'] +
                                                               distilled_features['digital_divide_rate'] +
                                                               distilled_features['linguistic_isolation_rate'] +
                                                               distilled_features['zero_car_rate'] +
                                                               distilled_features['low_income_pct']
                                                       ) / 5  # Average of vulnerability indicators

    # Economic metrics
    distilled_features['economic_vitality_index'] = (
                                                            distilled_features['employment_rate'] +
                                                            distilled_features['higher_education_rate'] +
                                                            distilled_features['high_income_pct'] +
                                                            1 - distilled_features['low_income_pct']  # Invert low income to make higher = better
                                                    ) / 4  # Average of economic indicators

    # Housing market dynamics
    distilled_features['housing_age_diversity'] = 1 - (
            (df['HA_AFT2010'] / df['HU_TOT']) ** 2 +
            (df['HA_90_10'] / df['HU_TOT']) ** 2 +
            (df['HA_70_90'] / df['HU_TOT']) ** 2 +
            (df['HA_40_70'] / df['HU_TOT']) ** 2 +
            (df['HA_BEF1940'] / df['HU_TOT']) ** 2
    )  # Diversity index for housing age

    # Housing type mix
    distilled_features['housing_type_diversity'] = 1 - (
            distilled_features['single_family_pct'] ** 2 +
            distilled_features['small_multifamily_pct'] ** 2 +
            distilled_features['large_multifamily_pct'] ** 2 +
            (df['HU_MOBILE'] / df['HU_TOT']) ** 2
    )  # Diversity index for housing types

    # Gentrification/displacement risk indicators
    distilled_features['gentrification_risk'] = (
                                                        distilled_features['rent_to_income'] +
                                                        distilled_features['home_value_to_income'] +
                                                        (df['RENT_GT2500'] / df['CASHRENT_HH']) +
                                                        (df['MED_RENT'] / df['MEDINC']) +
                                                        (1 - distilled_features['old_housing_pct'])  # Newer housing stock
                                                ) / 5  # Average of gentrification risk factors

    # Mobility dependence
    distilled_features['car_dependence_index'] = (
                                                         distilled_features['car_commute_rate'] +
                                                         distilled_features['multi_car_rate'] +
                                                         (1 - distilled_features['transit_use_rate']) +
                                                         (1 - distilled_features['active_commute_rate'])
                                                 ) / 4  # Average of car dependence indicators

    # Housing mismatch indicators
    distilled_features['housing_size_mismatch'] = abs(
        distilled_features['avg_household_size'] -
        (df['BR_0_1'] * 1.5 + df['BR_2'] * 2 + df['BR_3'] * 3 + df['BR_4'] * 4 + df['BR_5'] * 5) / df['HU_TOT']
    )  # Difference between avg household size and avg bedrooms

    # Family structure indicators
    distilled_features['non_traditional_hh_rate'] = (df['CT_NONFAM_HH'] + df['CT_SP_WCHILD']) / df['TOT_HH']

    # Assessed value per capita
    if 'TOT_EAV' in df.columns:
        distilled_features['eav_per_capita'] = df['TOT_EAV'] / distilled_features['recent_population']
        distilled_features['residential_eav_per_capita'] = df['RES_EAV'] / distilled_features['recent_population']
        distilled_features['commercial_eav_per_capita'] = df['CMRCL_EAV'] / distilled_features['recent_population']

    # Segregation/integration indicators
    if all(col in df.columns for col in ['WHITE', 'BLACK', 'HISP', 'ASIAN']):
        max_group = df[['WHITE', 'BLACK', 'HISP', 'ASIAN']].max(axis=1)
        distilled_features['segregation_index'] = max_group / df[['WHITE', 'BLACK', 'HISP', 'ASIAN']].sum(axis=1)

    # Community resources
    distilled_features['institutional_land_ratio'] = df['INSTperc'] if 'INSTperc' in df.columns else np.nan
    distilled_features['resource_access_index'] = (
                                                          distilled_features['institutional_land_ratio'] +
                                                          distilled_features['internet_access_rate'] +
                                                          (1 - distilled_features['zero_car_rate']) +
                                                          distilled_features['transit_use_rate']
                                                  ) / 4  # Average of resource access indicators

    # Economic diversity
    distilled_features['income_diversity'] = 1 - (
            (df['INC_LT_25K'] / df['TOT_HH']) ** 2 +
            (df['INC_25_50K'] / df['TOT_HH']) ** 2 +
            (df['INC_50_75K'] / df['TOT_HH']) ** 2 +
            (df['INC_75_100K'] / df['TOT_HH']) ** 2 +
            (df['INC_100_150K'] / df['TOT_HH']) ** 2 +
            (df['INC_GT_150'] / df['TOT_HH']) ** 2
    )  # Diversity index for income distribution

    # Handle any potential division by zero or missing values
    distilled_features = distilled_features.replace([np.inf, -np.inf], np.nan)

    # Remove columns with NaN values
    distilled_features = distilled_features.dropna(axis=1)


    return distilled_features



