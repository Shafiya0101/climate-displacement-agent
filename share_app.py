"""Launch the Gradio interface with a public share link."""
from gradio_app import demo

demo.queue(max_size=8).launch(share=True)
