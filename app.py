from flask import Flask, render_template, request, redirect, url_for
from flask_mysqldb import MySQL
import openai
import nbformat
import json
import ast
import os
app = Flask(__name__)

# Configure MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_PORT'] = 3307
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'mlops'
mysql = MySQL(app)

# Define function to parse notebook
def parse_colab_notebook(notebook_path):
    # Your existing parsing logic here...
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    cells = notebook_content['cells']

    parsed_cells = []

    for cell in cells:
        cell_info = {'code': '', 'outputs': []}

        if cell['cell_type'] == 'code' or cell['cell_type']=='markdown':
            cell_info['code'] = cell['source']

            if 'outputs' in cell:
                for output in cell['outputs']:
                    output_data = {}

                    if 'data' in output and 'text/plain' in output['data']:
                        output_data['text'] = output['data']['text/plain']
                    elif 'text' in output:
                        output_data['text'] = output['text']

                    cell_info['outputs'].append(output_data)

        parsed_cells.append(cell_info)

    return parsed_cells
@app.route('/')
def home():
    return render_template('home.html')

# Home page with form to input hypothesis
@app.route('/create_hypothesis')
def index():
    return render_template('create_hypothesis.html')

# Route to save hypothesis
@app.route('/save_hypothesis', methods=['POST'])
def save_hypothesis():
    # Get data from form
    hypothesis = request.form['hypothesis']
    assumptions = request.form['assumptions']
    results = request.form['results']
    
    # Save hypothesis to MySQL database
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO hypotheses (hypothesis, assumptions, results) VALUES (%s, %s, %s)", (hypothesis, assumptions, results))
    mysql.connection.commit()
    
    # Fetch the auto-generated hypothesis_id
    cur.execute("SELECT LAST_INSERT_ID()")
    hypothesis_id = cur.fetchone()[0]
    
    cur.close()
    
    # Redirect to create_experiment page with hypothesis_id
    return redirect(url_for('create_experiment', hypothesis_id=hypothesis_id))

# Page to create experiment
@app.route('/create_experiment')
def create_experiment():
    # Fetch all available hypotheses from database
    cur = mysql.connection.cursor()
    cur.execute("SELECT hypothesis_id, hypothesis FROM hypotheses")
    hypotheses = cur.fetchall()
    cur.close()
    
    return render_template('create_experiment.html', hypotheses=hypotheses)

# Route to handle uploaded notebook and run experiment
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    hypothesis_id = request.form['hypothesis_id']
    notebook_file = request.files['notebook_file']
    
    # Save uploaded notebook
    notebook_path = f"experiments/{notebook_file.filename}"
    notebook_file.save(notebook_path)
    
    # Parse the notebook
    parsed_cells = parse_colab_notebook(notebook_path)
    
    # Your existing GPT-3 code here...
    openai.api_key = "Your API KEY"
    blueprint='{"model_type": "string", "hyperparameters": {"parameter1": "value1"}, "input_attributes": {"attribute1": {"type": "string", "range": [category1, category2, ...], "attribute2": {"type": "int", "range": "numeric - numeric"}}, "target_attributes": {"target_attribute1": {"type": "string"}}, "results": {"test_accuracy": percent, "train_accuracy": percent}}'
    prompt = f"{parsed_cells}\nusing the above code and output for each cell, give your experiment analysis in the following format as a json object - {blueprint} -your response must be only the json object and nothing else. If there is any information on the split between test and train from the code, include those details in hyperparameters. The attribute type could be any of int, float, categorical, epoch, etc(should be a valid python datatype). If the attribute is of string or categorical type, include all possible categories in range as an array. Every value to an attribute must be enclosed within double quotes."
    #print(prompt)

    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",  # Choose the appropriate engine
        prompt=prompt,
        max_tokens=1000,  # Adjust based on your needs
        stream=False
    )

    # Extract relevant information from the GPT-3 response
    gpt3_output = response.choices[0].text
    
    gpt3_output = gpt3_output.replace('\n', ' ').replace('\r', '')
    #exp_details = json.loads(gpt3_output[0])
    exp_details=eval(gpt3_output)
    #exp_id exp hyp_id model hyperpara inp tar res
    model_type=exp_details['model_type']
    hyperparameters=exp_details['hyperparameters']
    input_attributes=exp_details['input_attributes']
    target_attributes=exp_details['target_attributes']
    results=exp_details['results']
    print(exp_details)
    return render_template('experiment_results.html', experiment="", hypothesis_id=hypothesis_id, model_type=model_type, hyperparameters=hyperparameters, input_attributes=input_attributes, target_attributes=target_attributes, results=results, notebook_path=notebook_path)


# Route to save experiment details to the experiments registry
@app.route('/save_experiment', methods=['POST'])
def save_experiment():
    # Get experiment details from the form
    hypothesis_id = request.form['hypothesis_id']
    experiment=request.form['experiment']
    model_type = request.form['model_type']
    hyperparameters = request.form['hyperparameters']
    input_attributes = request.form['input_attributes']
    target_attributes = request.form['target_attributes']
    results = request.form['results']
    notebook_path=request.form['notebook_path']
    
    # Save experiment details to the experiments registry in the MySQL database
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO experiments (experiment, hypothesis_id, model_type, hyperparameters, input_attributes, target_attributes, results, notebook_path) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (experiment, hypothesis_id, model_type, hyperparameters, input_attributes, target_attributes, results, notebook_path))
    mysql.connection.commit()
    cur.close()
    
    return "Experiment details saved successfully!"

# Route to display all hypotheses and their experiments
@app.route('/view_experiments', methods=['GET'])
def view_experiments():
    # Fetch all hypotheses and their experiments from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT hypotheses.hypothesis_id, hypotheses.hypothesis, experiments.experiment_id, experiments.experiment, experiments.model_type, experiments.results, experiments.notebook_path FROM hypotheses LEFT JOIN experiments ON hypotheses.hypothesis_id = experiments.hypothesis_id")
    hypothesis_experiments = cur.fetchall()
    cur.close()

    return render_template('view_experiments.html', hypothesis_experiments=hypothesis_experiments)

# Route to register a model from the selected experiment
@app.route('/register_model', methods=['POST', 'GET'])
def register_model():
    # Get data from the form
    hypothesis_id = request.form['hypothesis_id']
    experiment_id = request.form['experiment_id']
    model_name = request.form.get('model_name')
    model_file = request.files['model_file']
    # Fetch experiment details based on the experiment ID
    cur = mysql.connection.cursor()
    cur.execute("SELECT model_type, hyperparameters, input_attributes, target_attributes, results FROM experiments WHERE experiment_id = %s", (experiment_id,))
    experiment_details = cur.fetchone()
    cur.close()
    print(model_name)
    if experiment_details:
        model_type, hyperparameters, input_attributes, target_attributes, results= experiment_details
        models_folder = os.path.join(os.getcwd(), 'models')  # 'models' folder in the current directory
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        model_path = os.path.join(models_folder, model_file.filename)
        model_file.save(model_path)
        # Save the selected model to the model registry in the MySQL database
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO model_registry (experiment_id, model_name, model_type, hyperparameters, input_attributes, target_attributes, results, model_path) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", (experiment_id, model_name, model_type, hyperparameters, input_attributes, target_attributes, results, model_path))
        mysql.connection.commit()
        cur.close()

        return redirect(url_for('select_model'))
    else:
        return "Experiment details not found!"

@app.route('/select_model', methods=['GET', 'POST'])
def select_model():
    # Retrieve registered models from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT model_id, model_name FROM model_registry")
    registered_models = cur.fetchall()
    cur.close()

    if request.method == 'POST':
        # Get the selected model ID from the form
        selected_model_id = request.form.get('model_id')

        # Redirect to the deployment route with the selected model ID
        return redirect(url_for('deploy_model', model_id=selected_model_id))

    return render_template('select_model2.html', registered_models=registered_models)

# Existing imports and configurations...

@app.route('/deploy_model/<int:model_id>')
def deploy_model(model_id):
    # Fetch model details from the database based on the model ID
    cur = mysql.connection.cursor()
    cur.execute("SELECT model_name FROM model_registry WHERE model_id = %s", (model_id,))
    model_name = cur.fetchone()
    cur.close()

    if model_name:
        # Redirect to the FastAPI app for model deployment
        return redirect(f"http://127.0.0.1:8000/predict_page/{model_name[0]}")  # Update the URL with the FastAPI app's address
    else:
        return "Model not found!"

# Other routes and app configurations remain the same...

if __name__ == '__main__':
    app.run(debug=True)