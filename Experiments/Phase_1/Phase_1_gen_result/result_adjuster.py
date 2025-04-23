# Creating DataFrames for tabular comparison, this file is due to mismtach of results between falcon and the other 2 models

# Table for 200 Questions (Mistral vs. Legal LLaMA)
comparison_200 = pd.DataFrame({
    "Model": ["Mistral", "Legal LLaMA"],
    "Semantic Similarity": semantic_similarity_200,
    "BERT Score": bert_score_200,
    "ROUGE-L F1": rouge_l_f1_200,
    "Evaluation Time (sec)": eval_time_200
})

# Table for First 50 Questions (Mistral vs. Legal LLaMA vs. Falcon)
comparison_50 = pd.DataFrame({
    "Model": ["Mistral", "Legal LLaMA", "Falcon"],
    "Semantic Similarity": semantic_similarity_50,
    "BERT Score": bert_score_50,
    "ROUGE-L F1": rouge_l_f1_50,
    "Evaluation Time (sec)": eval_time_50
})

# Display the tables
tools.display_dataframe_to_user(name="Performance Comparison (200 Questions)", dataframe=comparison_200)
tools.display_dataframe_to_user(name="Performance Comparison (First 50 Questions)", dataframe=comparison_50)
