# MLDS_414_Text_Analytics_Movie_Genre

[TODO: short description]

## Project Structure

[TODO]

## Project Goals

[TODO]

## Models

[TODO]

## Run the Project

### 1. Install Dependencies
```bash
conda create -n 414-final-proj python=3.9
conda activate 414-final-proj
pip install -r requirements.txt
```

### 2. Train the Model [TODO: add in all models]
```bash
python DistilBERT_genre_training.py

# optional for inference
python DistilBERT_genre_inference.py --text "A man must fight through zombies to protect his family."
```

### 3. Start the Backend and Frontend
```bash
python backend/app.py
streamlit run streamlit_app.py
```

## Example Use Case

Paste the movie description to the app UI:

> Four years after the catastrophic events that led to the destruction of Isla Nublar, the world is now a drastically changed place. Dinosaurs, once confined to the isolated ecosystems of the island, are no longer simply a distant memory or an attraction in a theme park—they are part of the global landscape. These ancient creatures roam free, inhabiting diverse environments and hunting alongside humans in an uneasy coexistence that constantly teeters on the edge of chaos. As dinosaurs find their place in a rapidly evolving ecosystem, the delicate balance between mankind and prehistoric predators becomes more fraught. The human race must confront its place on a planet that no longer belongs solely to them. With species both old and new adapting to the modern world, the consequences of mankind’s previous attempts to control, exploit, and manipulate the natural world come into sharp focus. This precarious harmony challenges every assumption about the survival of the human race as new threats arise, and old fears are reignited. Against this backdrop, the future of humanity will be determined, as it’s no longer about simply surviving alongside these magnificent creatures, but about understanding whether humans can coexist—or if they will be relegated to the role of prey. The age-old question lingers: who will remain the apex predator in a world where mankind now shares the Earth with some of history's most fearsome and intelligent creatures?

Output: 

> Action

![frontend user interface](assets/frontend_ui.png)

## Evaluation

```json
{
    "eval_loss": 0.6107685565948486,
    "eval_accuracy": 0.745,
    "eval_runtime": 0.548,
    "eval_samples_per_second": 364.965,
    "eval_steps_per_second": 9.124,
    "epoch": 5.0
}
```

## Authors

[TODO]