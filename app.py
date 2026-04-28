from flask import Flask, render_template, request
import pandas as pd
import pickle
print("APP STARTING...")
app = Flask(__name__)

# Load trained model
model = pickle.load(open('fraud_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    print("UPLOAD FUNCTION CALLED")

    try:
        use_local = request.form.get('use_local')
        if use_local == 'yes':
            df = pd.read_csv('demo_dataset.csv')
            print("LOCAL FILE READ SUCCESS")
        else:
            if 'file' not in request.files:
                return "No file part"
            file = request.files.get('file')
            if file is None or file.filename == '':
                return "No file uploaded", 400
            df = pd.read_csv(file)
            print("FILE READ SUCCESS")
        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)

        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        scaled_data = scaler.transform(df)
        predictions = model.predict(scaled_data)

        print("PREDICTION DONE")   # 👈 ADD THIS

        df['Fraud Result'] = predictions

        fraud_count = int((predictions == 1).sum())
        safe_count = int((predictions == 0).sum())
        total = len(predictions)
        fraud_percent = round((fraud_count / total) * 100, 2)

        fraud_rows = df[df['Fraud Result'] == 1].head(5)
        safe_rows = df[df['Fraud Result'] == 0].head(5)
        sample = pd.concat([fraud_rows, safe_rows])
        table = sample[['Amount', 'Fraud Result']].to_html(index=False)

        return render_template(
            'index.html',
            table=table,
            fraud_count=fraud_count,
            safe_count=safe_count,
            fraud_percent=fraud_percent
        )

    except Exception as e:
        print("ERROR:", e)
        return render_template('index.html', error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)