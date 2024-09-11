import gradio as gr
from generate_prediction import generate_prediction

theme = gr.themes.Default(
    primary_hue="stone",
    secondary_hue="blue",
    neutral_hue="zinc",
    spacing_size="md",
    text_size="md",
    font=[gr.themes.GoogleFont("IBM Plex Mono"), "system-ui"]
)

def predict_image(image):
    # Save the uploaded image to /image.jpeg
    image_path = "./image.jpeg"
    image.save(image_path)

    # Call your model's prediction function
    prediction = generate_prediction(image_path)  # Assuming load_and_predict function exists in your model file

    return prediction

with gr.Blocks(theme=theme) as demo:
    # DEFINE COMPONENTS
    gr.Markdown("# MindReader Quantum")

    # Uploading the image input
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        output_label = gr.Label(label="Prediction")

    # Button to submit and show the prediction
    with gr.Row():
        submit_btn = gr.Button("Submit")

    submit_btn.click(fn=predict_image, inputs=image_input, outputs=output_label)

demo.launch()
