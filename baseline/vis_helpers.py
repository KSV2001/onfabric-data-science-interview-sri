import pandas as pd
import matplotlib.pyplot as plt

def visualize_theme_timeline(model):
    """
    Visualize latent themes over time  for the baseline model.

    Parameters
    ----------
    model : BaselineUserModel
        Trained user model with theme assignments.
    """
    rows = [
        {
            "timestamp": e.timestamp,
            "theme_id": e.theme_id,
            "description": e.description,
        }
        for e in model.events
        if e.theme_id is not None
    ]

    if not rows:
        print("No themed events to visualize.")
        return

    df = pd.DataFrame(rows)

    plt.figure(figsize=(14, 4))
    scatter = plt.scatter(
        df["timestamp"],
        df["theme_id"],
        c=df["theme_id"],
        cmap="tab20",
        s=12,
        alpha=0.8,
    )
    plt.xlabel("Time")
    plt.ylabel("Theme ID")
    plt.title("Latent Themes Over Time (Baseline)")
    plt.show()
