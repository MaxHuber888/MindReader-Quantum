import gradio as gr
from helpers import load_video_from_url, detect_deepfake

theme = gr.themes.Default(
    primary_hue="stone",
    secondary_hue="blue",
    neutral_hue="zinc",
    spacing_size="md",
    text_size="md",
    font=[gr.themes.GoogleFont("IBM Plex Mono"), "system-ui"]
)

callback = gr.CSVLogger()

with gr.Blocks(theme=theme) as demo:
    # DEFINE COMPONENTS

    # Text box for inputting Youtube URL
    urlInput = gr.Textbox(
        label="YOUTUBE VIDEO URL",
        value="https://www.youtube.com/watch?v=BmrUJhY9teE"
    )

    # Button for downloading the video and previewing sample frames
    loadVideoBtn = gr.Button("Load Video")

    # Text box for displaying video title
    videoTitle = gr.Textbox(
        label="VIDEO TITLE",
        lines=1,
        interactive=False
    )

    # Image Gallery for previewing sample frames
    sampleFrames = gr.Gallery(
        label="SAMPLE FRAMES",
        elem_id="gallery",
        columns=[3],
        rows=[1],
        object_fit="contain",
        height="auto"
    )

    # Button for generating video prediction
    predVideoBtn = gr.Button(value="Classify Video", visible=False)

    # Label for displaying prediction
    predOutput = gr.Label(
        label="DETECTED LABEL (AND CONFIDENCE LEVEL)",
        num_top_classes=2,
        visible=False
    )

    # Button for flagging the output
    flagBtn = gr.Button(value="Flag Output", visible=False)

    # DEFINE FUNCTIONS
    # Load video from URL, display sample frames, and enable prediction button
    loadVideoBtn.click(fn=load_video_from_url, inputs=[urlInput], outputs=[videoTitle, sampleFrames, predVideoBtn, predOutput])

    # Generate video prediction
    predVideoBtn.click(fn=detect_deepfake, outputs=[predOutput, flagBtn])

    # Define flag callback
    callback.setup([urlInput], "flagged_data_points")

    # Flag output
    flagBtn.click(fn=lambda *args: callback.flag(args), inputs=[urlInput], outputs=None)

demo.launch()
