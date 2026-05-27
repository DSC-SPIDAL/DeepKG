# =============================================================================
# V83: Schema (URL Separation and Confidence Updates)
# Features:
# 1. Added 'Other URL' and 'Other URL (C)'.
# 2. Added 'Link to Data (Actual Source) (C)'.
# 3. Enforced strict rules against Google Scholar/Vertex AI in RAG prompts.
# =============================================================================

KB_SCHEMA_VERSION = "24.04"

# DEFINITION OF TABS AND COLUMNS
KB_SCHEMA = {
    "Jobs": [
        "JobID", "ProjectID", "Mode", "Start_Time", "End_Time", "Duration_Sec",
        "Status", "Items_Found", "Operational_Parameters", "JOB_COMMENT"
    ],
    "Datasets": [
        "DatasetID", "Canonical Name", "Variant Name", "Aliases", "Type",
        "Num Time Points", "Num Time Points (C)",
        "Num Locations/Series", "Num Locations/Series (C)",
        "Total Variables", "Total Variables (C)",
        "Variables per Location", "Variables per Location (C)",
        "Frequency", "Frequency (C)",
        "Domain", "Domain (C)",
        "Description",
        "Primary Creator",
        "Primary URL", "Primary URL (C)",
        "Link to Data (Actual Source)", "Link to Data (Actual Source) (C)",
        "Other URL", "Other URL (C)",
        "License", "Overall Confidence",
        "Job_Created", "Date_Created", "Project_Created",
        "Job_Updated", "Date_Updated", "Project_Updated"
    ],
    "Projects": [
        "ProjectID", "Name", "Description", "Type",
        "Last Analyzed", "Source URL"
    ],
    "Project_Dataset_Link": [
        "LinkID", "ProjectID", "DatasetID", "Actual Data URL Used",
        "Data Preparation Comments", "CitationID", "Assignment Confidence",
        "Link_Date", "Linked_By_Job"
    ],
    "Datasets_Quarantined": [
        "DatasetID", "Canonical Name", "Variant Name", "Aliases", "Type",
        "Num Time Points", "Num Time Points (C)",
        "Num Locations/Series", "Num Locations/Series (C)",
        "Total Variables", "Total Variables (C)",
        "Variables per Location", "Variables per Location (C)",
        "Frequency", "Frequency (C)",
        "Domain", "Domain (C)",
        "Description",
        "Primary Creator",
        "Primary URL", "Primary URL (C)",
        "Link to Data (Actual Source)", "Link to Data (Actual Source) (C)",
        "Other URL", "Other URL (C)",
        "License", "Overall Confidence",
        "Job_Created", "Date_Created", "Project_Created",
        "Job_Updated", "Date_Updated", "Project_Updated"
    ],
    "Links_Quarantined": [
        "LinkID", "ProjectID", "DatasetID", "Actual Data URL Used",
        "Data Preparation Comments", "CitationID", "Assignment Confidence",
        "Link_Date", "Linked_By_Job"
    ],
    "Datasets_Duplicates_Log": [
        "DatasetID (Dropped)", "DatasetID (Kept)", "Reason", "Merge Notes", "Dropped Name", "Kept Name"
    ],
    "Citations": [
        "CitationID", "Title", "Authors", "Venue", "Year", "DOI", "URL",
        "Full Citation Text"
    ],
    "WebLinks": ["WebLinkID", "URL", "Resource Type", "Description"]
}

# Internal Agent Schema (Mappings for RAG)
CATALOG_SCHEMA = {
    # Identifiers
    "Dataset Name": {"description": "The specific name of the dataset variant (e.g., ETTh1)."},
    "Canonical Name": {
        "description": "The standardized, primary group name (e.g., ETT, M4, PeMS).",
        "query": "What is the primary group, family, or canonical name for the {name} dataset? (e.g. if 'ETTh1', answer 'ETT'; if 'M4 Hourly', answer 'M4'). If it has no group, return the dataset name itself."
    },
    "Aliases": {"description": "Other names or variants related to the canonical name."},
    "Type": {
        "description": "The classification of the entity.",
        "query": "Classify the entity '{name}'. Is it a [Real-World Dataset | Synthetic Dataset | Synthetic Generator | Augmentation Tool | Evaluation Script | Collection | Provider]? Reply STRICTLY with exactly ONE of these options."
    },

    # Assignment Confidence
    "Assignment Confidence": {"description": "Confidence (C_relevance) that the dataset is relevant to the current project context."},
    "Assignment Rationale": {"description": "Justification for the relevance of the dataset to the project."},

    # Core Metadata
    "Domain": {"description": "The area of application (e.g., Finance, Energy).", "query": "What is the domain or area of application for the {name} dataset?"},
    "Detailed Description": {"description": "A single-line summary.", "query": "Provide a detailed, single-line description of the contents and context of the {name} dataset."},

    # Time Series Specifications
    "Time interval between points": {"description": "Frequency (e.g., 1 hour, 15 minutes).", "query": "What is the frequency or time interval between successive points specifically for the version of the {name} dataset used in this context?"},
    "Number of Time Points": {"description": "Length or total samples.", "query": "What is the total length (number of time points or samples at a given location) specifically for the version of the {name} dataset used in this context? Must be a number or range."},
    "Number of Locations/Series": {"description": "The number of distinct spatial units or clients.", "query": "How many distinct locations, clients, or independent time series are included in the specific version of the {name} dataset used here? Must be a number or range."},
    "Variables per Location": {
        "description": "The number of features measured at each location.",
        "query": "How many distinct variables/features/channels are measured at EACH single location/time-step for the {name} dataset? CRITICAL: Do NOT count the number of time points (length). If the shape is (1000, 6), the answer is 6."
    },
    "Total Variables": {
        "description": "The total dimensionality.",
        "query": "What is the total number of variables or features for the {name} dataset? CRITICAL: Do NOT count the number of time steps (length). If the dataset has 1000 time steps and 5 variables, the answer is 5. If strictly univariate, answer 1."
    },

    # Grounding & Provenance (URL UPDATES HERE)
    "Primary Source Repository": {
        "description": "The official originator or primary host.",
        "query": "CRITICAL: Identify the official institution or repository that *originated* the {name} dataset. Provide the Name."
    },
    "Primary URL": {
        "description": "Home Page for the dataset.",
        "query": "What is the official Home Page URL for the {name} dataset? MUST be a true URL. ABSOLUTELY NO links to Google Scholar, Vertex AI, or generic search results. If multiple, separate by commas."
    },
    "Link to Data (Actual Source)": {
        "description": "Link(s) directly to the DATASET LOCATION from which the data files can be downloaded.",
        "query": "Provide the direct download URL(s) or repository link(s) (e.g., GitHub, UCI, PhysioNet, Kaggle) for the actual data files of the {name} dataset. MUST be a true URL. ABSOLUTELY NO Google Scholar or Vertex AI links. If multiple, separate by commas."
    },
    "Other URL": {
        "description": "Academic papers, supplementary GitHub repositories, and any other useful links.",
        "query": "Provide academic paper URLs, documentation, or other supplementary links for the {name} dataset. MUST be true URLs. ABSOLUTELY NO Google Scholar or Vertex AI links. If multiple, separate by commas."
    },
    "Comments on Data Preparation": {"description": "Notes on modifications or preprocessing.", "query": "Describe any modifications, preprocessing, or version differences for the {name} dataset compared to its primary source. If standard, state 'Standard version'."},

    "Project Citations": {"description": "Academic papers associated with the project."},
    "Project WebLinks": {"description": "Web resources (GitHub, Homepages) associated with the project."},
}

GROUNDING_FIELDS = ["Primary Source Repository", "Primary URL", "Link to Data (Actual Source)", "Other URL", "Comments on Data Preparation"]
EXTRACTED_FIELDS = [field for field, definition in CATALOG_SCHEMA.items() if 'query' in definition]
ASSIGNMENT_FIELDS = ["Dataset Name", "Assignment Confidence", "Assignment Rationale"]

DDI_INSPECTABLE_FIELDS = [
    "Variables per Location", "Total Variables",
    "Number of Locations/Series", "Number of Time Points"
]

MISSING_DATA_PLACEHOLDERS = {
    "", "[missing]", "unknown", "n/a", "not found", "not specified",
    "not available", "not mentioned", "none", "missing", "nan", "null",
    "[implausible]", "[error]", "[stability_error]", "[timeout_error]", "[server_error]", "[retrieval_error]",
    "[extraction_error: dict_returned]", "[extraction_error: complex_string]"
}

PLAUSIBILITY_THRESHOLDS = {
    "Variables per Location": {"min": 1, "max": 50000},
    "Number of Locations/Series": {"min": 1, "max": 10000000},
    "Number of Time Points": {"min": 1, "max": 100000000},
    "Total Variables": {"min": 1, "max": 1000000000}
}
print("✅ deepcollector/config/schema.py written (V83: URL Separation).")