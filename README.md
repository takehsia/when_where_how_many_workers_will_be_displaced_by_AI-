# When, Where, and How many workers will be displaced by AI?
# Simulation for AI-driven job displacement considering digital infrastructure constraints

This repository contains the Python-based simulation framework used to
investigate the potential implications of LLMs on the
U.S. labor market. The model integrates technical AI exposure scaling with digital infrastructure constraints (cloud computing and specialized
software usage) to project job displacement trajectories from 2023 to 2055.

The analysis accounts for heterogeneity across industries and firm sizes (SMEs
vs. Large Corporations), treating digital infrastructure as a critical governor
of AI deployment speed.
If you run the code, please download all files and keep the structure of folder. 

# Code Structure and File Descriptions

# 1. Main Analysis (Simulation)
You can get all table and figures included the paper with running this code.
File: Submit_260428_Main_Analysis.ipynb
  - Description: The core simulation engine. It runs a Monte Carlo simulation
    (1,000 replicates) combining AI evolution, digital infra diffusion, and
    historical industry growth. It produces year-by-year projections of job
    displacement and automatable task shares. You can get every result if you run this code. 
  - Input: Cleaned staffing patterns, occupation AI ratings, DX adoption speeds,
    and historical SUSB regression data.
  - Output:
      - Final Impact Analysis CSV (longitudinal results).
      - Macro Employment Summary CSV.
      - Visualizations: Strategic Map (Snapshot),
        Industry Impact Histograms,and Trajectory Line Graphs.

# 2. Supplementary Materials 
# 2.1. AI Evolution Estimation
Folder:eci-public-main
File: fit_eci_submit.ipynb

  - Description: Analyzes the progress of AI capabilities. It uses Item Response
    Theory to fit the ECI scores and then applies an exponential growth
    model to determine the mean annual expansion rate (mu) and standard
    deviation (sigma) of AI progress.
    Please run two code 
  - Input: eci_benchmarks.csv
  - Output: eci_scores.csv, processed_capability_data.csv, and the statistical
    parameters used for the Monte Carlo simulation.

# 2.2. Digital Infrastructure Development Speed
File: submit_260419_DII_speed_parameter.ipynb

  - Description: Computes the "d" parameter (diffusion speed) for digital
    infrastructure. It compares adoption rates of cloud and software
    between 2018 and 2022 using NSF data and calculates the industry-specific
    adoption odds-ratio.
  - Input: BLS staffing patterns and NSF Excel tables (Cloud/Software adoption
    and size gaps).
  - Output: DX_Adoption_Speed_K_6_4.csv


# 2.3. Data Integration and Cleaning
This is for replication of data integration and cleaning.
You can run the main analysis code without running the data integration and cleaning code, since the cleaned datasets have already been saved in the repository.

Note: Since some original detests are so large and require a high-performance computing system to process. If you run this code in your local environment, it will not work well.
If high performance computing systems are not available, please use the cleaned datasets, that have already been in the repository.

Folder: Data_integration_cleaning
File: submit_260402_Data_integration_and_cleaning.ipynb

  - Description: This is the primary preprocessing script. It handles the raw
    Excel files from BLS and SUSB, standardizes NAICS codes, removes broad
    aggregate categories to retain granular 6-digit industry data, and creates a
    longitudinal panel (2012-2022) for historical trend estimation.
  - Input: Raw BLS (all_data_M_2024.xlsx) and multi-year SUSB Excel files.
  - Output:
      - bls_staffing_pattern_exclude_multipler1.csv
      - Employment_and_Payroll_by_industry_by_companysize_susb_united_states_only.csv
      - SUSB_Final_Regression_Data_a_h_i_Only.csv

Data Sources

The simulation utilizes several external datasets:

1.  U.S. Bureau of Labor Statistics (BLS) - OEWS Data

      - Description: Occupational Employment and Wage Statistics. Provides
        staffing patterns (the mix of occupations within each industry).
      - Source: Bureau of Labor Statistics.

2.  U.S. Census Bureau - Statistics of U.S. Businesses (SUSB)

      - Description: Annual data on the number of firms, employment, and payroll
        by industry and enterprise size. Used for historical trend analysis and
        firm-size weighting.
      - Source: U.S. Census Bureau.

3.  National Science Foundation (NSF) - Annual Business Survey (ABS)

      - Description: Data on technology adoption (Cloud and Software) across
        U.S. industries.
      - Source: NSF/NCES Annual Business Survey.

4.  Epoch AI - Epoch Capabilities Index (ECI)

      - Description: A dataset of AI model performance over time. Used to
        estimate the speed of AI capability growth.
      - Source: Epoch AI.

5.  Eloundou et al. (2023) - AI Exposure Data

      - Description: Occupation-level ratings of potential exposure to LLM
        capabilities.
      - Source: "GPTs are GPTs: An Early Look at the Labor Market Impact
        Potential of Large Language Models".


Requirements

This project requires Python 3.8+ and the following libraries:

  - pandas
  - numpy
  - matplotlib
  - seaborn
  - statsmodels
  - scikit-learn
  - openpyxl

You can install the dependencies via: pip install pandas numpy matplotlib
seaborn statsmodels scikit-learn openpyxl

