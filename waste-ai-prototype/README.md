# AI-Assisted Predictive Waste Analytics Prototype

This project is a Streamlit-based prototype for smart waste collection decisions, manager approval alerts, and peak waste prediction.

## What This Prototype Shows

- Predicts peak waste conditions from historical smart-bin style data
- Recommends pickup only when bins are near full or overflow is likely
- Flags records that need manager approval
- Displays a live dashboard for mentor demo and presentation

## Project Structure

```text
waste-ai-prototype/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── scripts/
│   └── prepare_data.py
└── src/
    ├── config.py
    ├── preprocess.py
    ├── features.py
    ├── rules.py
    ├── train_model.py
    └── utils.py
```

## Setup

1. Create a virtual environment if you want one.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the raw dataset into the project:

```bash
python scripts/prepare_data.py
```

By default, this copies from:

`D:\Downloads\final_large_dataset.xls`

If your source path changes, update `DEFAULT_SOURCE_DATASET` in `src/config.py`.

4. Train the model and generate the processed dataset:

```bash
python -m src.train_model
```

5. Start the dashboard:

```bash
streamlit run app.py
```

## Prototype Logic

### Pickup Decision

- `Dispatch Immediately` if fill level is at least 90% or overflow alert is true
- `Schedule Pickup` if fill level is at least 80% and a truck is available
- `Pickup Required but No Vehicle` if fill level is at least 80% but no truck is available
- `No Pickup Needed` otherwise

### Manager Approval

- Approval is required for hazardous waste
- Approval is also required for high pickup cost records

### Prediction Target

The prototype predicts a derived peak-day condition using enriched waste and operational features.

## Suggested Demo Flow

1. Open the dashboard and explain the objective
2. Show how the system identifies pickup-required bins
3. Show manager approval alerts
4. Show predicted peak waste records
5. Explain how this can later be adapted to a company-specific industrial workflow

## Important Note

This is a prototype dataset pipeline built on public or simulated smart-bin style data. It is suitable for demonstration and model validation, but real industry deployment will require company-specific production and waste-operation data.
