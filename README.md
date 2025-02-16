# RAGE DeepSeek

```txt
DeepSeekRAGE/
├── src/
│   ├── memory.py
│   ├── logger.py
│   ├── openmind.py
│   └── locallama.py
├── gfx/
│   └── styles.css
├── memory/
│   ├── sessions/long_term_memory.json
│   ├── knowledge/
└── rage.py
```
# DeepSeekRAGE
Retrieval Augmented Generative Engine<br />
DeepSeek RAGE<br />

RAGE Retrieval Augmented Generative Engine is a dynamic engine designed to learn from context, injest and memory over time. While I have had the idea for sometime no working expression of RAGE has been created, until this weekend.

# Context-Aware Responses: 
By leveraging the continuously updated data and learning from past interactions, RAGE can understand and respond to nuances in user queries. This ability makes it particularly effective in scenarios where context heavily influences the nature of the response.
# Adaptive Response Generation: 
As RAGE evolves, it becomes more adept at predicting user needs and adjusting its responses accordingly, ensuring high relevance and personalization.

```python
git clone https://github.com/GATERAGE/DeepSeekRAGE
```
```python
python3.11 -m venv rage
```
```python
source rage/bin/activate
```
```python
pip install --no-cache-dir -r requirements.txt
```

```python
streamlit run rage.py
```
