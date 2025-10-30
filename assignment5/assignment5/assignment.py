{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5ba5b752",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\user\\anaconda3\\envs\\python\\Lib\\site-packages\\sklearn\\base.py:376: InconsistentVersionWarning: Trying to unpickle estimator DictVectorizer from version 1.6.1 when using version 1.5.1. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Pipeline loaded successfully from pipeline_v1.bin\n",
      "\n",
      "--- Scoring Result ---\n",
      "Record: {'lead_source': 'paid_ads', 'number_of_courses_viewed': 2, 'annual_income': 79276.0}\n",
      "Class 0 Probability: 0.4664\n",
      "Class 1 Probability: 0.5336\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\user\\anaconda3\\envs\\python\\Lib\\site-packages\\sklearn\\base.py:376: InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression from version 1.6.1 when using version 1.5.1. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n",
      "c:\\Users\\user\\anaconda3\\envs\\python\\Lib\\site-packages\\sklearn\\base.py:376: InconsistentVersionWarning: Trying to unpickle estimator Pipeline from version 1.6.1 when using version 1.5.1. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# --- The record to be scored ---\n",
    "# Note: The model was trained with 'lead_source=NA' and other categorical features.\n",
    "# A dictionary is the expected input format for the first step (DictVectorizer).\n",
    "input_record = {\n",
    "    \"lead_source\": \"paid_ads\",\n",
    "    \"number_of_courses_viewed\": 2,\n",
    "    \"annual_income\": 79276.0\n",
    "}\n",
    "\n",
    "# 1. Define the file path\n",
    "pipeline_file = \"pipeline_v1.bin\"\n",
    "\n",
    "# 2. Load the pipeline using pickle\n",
    "try:\n",
    "    with open(pipeline_file, 'rb') as file:\n",
    "        pipeline = pickle.load(file)\n",
    "    print(f\"✅ Pipeline loaded successfully from {pipeline_file}\\n\")\n",
    "\n",
    "    # 3. Score the single record\n",
    "    # The pipeline expects a list of dictionaries, even for a single record.\n",
    "    # The pipeline's last step is a LogisticRegression (a classifier).\n",
    "    # .predict_proba() gives the probability for each class (0 and 1).\n",
    "    prediction_proba = pipeline.predict_proba([input_record])\n",
    "    \n",
    "    # 4. Extract and format the result\n",
    "    # The output is a 2D array: [[Prob_Class_0, Prob_Class_1]]\n",
    "    prob_class_0 = prediction_proba[0, 0]\n",
    "    prob_class_1 = prediction_proba[0, 1]\n",
    "    \n",
    "    print(\"--- Scoring Result ---\")\n",
    "    print(f\"Record: {input_record}\")\n",
    "    print(f\"Class 0 Probability: {prob_class_0:.4f}\")\n",
    "    print(f\"Class 1 Probability: {prob_class_1:.4f}\")\n",
    "\n",
    "except FileNotFoundError:\n",
    "    print(f\"❌ Error: The file '{pipeline_file}' was not found. Make sure it is in the same directory as the script.\")\n",
    "except Exception as e:\n",
    "    print(f\"❌ An error occurred during loading or prediction: {e}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "python",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
