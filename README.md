# AI-Assisted Predictive Waste Analytics Prototype

A Streamlit-based prototype for smart waste collection that predicts peak waste conditions, optimizes pickup decisions, and flags manager approvals.

## Key Features
- Predicts peak waste using historical smart-bin data
- Recommends pickups only when necessary (near full/overflow risk)
- Flags records requiring manager approval (hazardous or high cost)
- Interactive dashboard for demo and visualization

## Workflow
1. Prepare dataset → `prepare_data.py`
2. Train model → `train_model.py`
3. Launch dashboard → `streamlit run app.py`

## Core Logic
- Dispatch Immediately: Fill ≥ 90% or overflow
- Schedule Pickup: Fill ≥ 80% + truck available
- No Vehicle: Fill ≥ 80% but no truck
- No Action: Otherwise

## Note
Prototype uses simulated/public data—real deployment requires company-specific datasets.

## Dashboard Preview
<img width="1898" height="481" alt="image" src="https://github.com/user-attachments/assets/9021c8e1-f95d-4e67-8595-c2d48488bb34" />
<img width="1904" height="465" alt="image" src="https://github.com/user-attachments/assets/a2d04d35-e752-40d1-954f-f9940312e7b7" />
<img width="1898" height="151" alt="image" src="https://github.com/user-attachments/assets/a3eb75b8-f4d0-4045-a8f1-7d0863657531" />
<img width="1900" height="467" alt="image" src="https://github.com/user-attachments/assets/ce27e848-2260-446d-967a-82beee8ad929" />
<img width="1889" height="470" alt="image" src="https://github.com/user-attachments/assets/2564d564-956f-429d-a052-a58d5b4181d7" />
<img width="1895" height="436" alt="image" src="https://github.com/user-attachments/assets/a25490a5-cf08-4203-89a9-01cf4848bffb" />
<img width="1895" height="449" alt="image" src="https://github.com/user-attachments/assets/14111151-d5cc-4337-a8bc-703e37e4a86e" />
<img width="1897" height="460" alt="image" src="https://github.com/user-attachments/assets/3111131a-9384-4451-b0ee-cb144f5beaa9" />

