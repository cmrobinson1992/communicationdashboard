# SLTX Communications Dashboard (With NLP Classification)

**## This project delivers a flexible, user-friendly **interactive communication performance dashboard** for the Surplus Lines Stamping Office of Texas (SLTX) Operations Department. It integrates Microsoft Outlook and Teams data to quantify, visualize, and analyze communication activity by method (email or phone), direction, and content. Additionally, it implements a natural language processing (NLP) model to classify emails into categories to support decision-making and performance evaluation.
**NOTE: Because of sensitive information included in e-mails, an interactive version of the dashboard and the data set used is unavailable**

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
- [Dashboard Screenshots](#dashboard-screenshots)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

While SLTX historically emphasized metrics for data entry and policy audits, this project highlights **communication as a vital but underrepresented aspect of Policy Analyst performance**. 

By combining structured communication data and NLP-based email categorization, the dashboard provides management with actionable insights to improve transparency, responsiveness, and workload distribution.

---

## Features

- 📈 **Interactive Shiny Dashboard** in R:
  - Communication counts by method and direction
  - Time-series trends
  - User rankings by volume
  - Category summaries via tooltips and popovers
  - Toggle views for phone call counts and durations

- 🤖 **Email Classification Model:**
  - Trained on >10,000 emails
  - Categories include Filing Assistance, Technical Support, Audits, Reports, API Mapping, and Other
  - Supports prediction for real-time categorization

- 🧪 **Model Evaluation:**
  - Tested Multinomial Naive Bayes, SVM, and CNN classifiers
  - SVM with linear kernel selected for best accuracy and stability (~95% mean accuracy)

---

## Data Sources

The dashboard uses two primary data sources:

1. **Email Data:**
   - Extracted from Outlook shared folders (15,731 raw emails)
   - Cleaned and preprocessed in Python
   - Fields include date, subject, body, category, attachments, and direction

2. **Phone Data:**
   - Microsoft Teams operator and user call logs
   - Aggregated across multiple months and formatted for analysis

---

**Final Paper**

[(https://github.com/cmrobinson1992/communicationdashboard/blob/main/Practicum_Project-%20Final.pdf)]

Future Improvements
* Transition from static CSV ingestion to real-time APIs

* Enhance email classification performance on minority categories

* Automate user access controls and role-based dashboards

* Integrate live notification systems for threshold alerts

Contact
Christian Robinson
📧 christian.m.robinson@gmail.com


   
