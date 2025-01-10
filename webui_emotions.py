from pydantic import BaseModel
from typing import Optional
import gradio as gr
import os
os.environ["XDG_RUNTIME_DIR"] = "/tmp"
os.environ["AUDIODEV"] = "null"  # Suppress ALSA-related logs

# Your other imports
from code_for_webui.download_models_openxlab import download 
from code_for_webui.demo_EDTalk_A_using_predefined_exp_weights import Demo as Demo_EDTalk_A_using_predefined_exp_weights
from code_for_webui.demo_lip_pose import Demo as Demo_lip_pose

# Define Pydantic model for input validation
class InputData(BaseModel):
    source_image: str
    need_crop_source_img: bool
    audio_file: Optional[str]
    pose_video: Optional[str]
    need_crop_pose_video: bool
    exp_type: str
    face_sr: bool

    class Config:
        arbitrary_types_allowed = True


# Initialize demo instances
demo_lip_pose = Demo_lip_pose()
demo_EDTalk_A_using_predefined_exp_weights = Demo_EDTalk_A_using_predefined_exp_weights()

def run_inference(input_data: InputData):
    # Now use the input_data object to access input parameters
    source_path = input_data.source_image if input_data.source_image else ""
    audio_driving_path = input_data.audio_file if input_data.audio_file else ""
    pose_driving_path = input_data.pose_video if input_data.pose_video else "test_data/pose_source1.mp4"
    exp_type = input_data.exp_type
    need_crop_source_img = input_data.need_crop_source_img
    need_crop_pose_video = input_data.need_crop_pose_video
    face_sr = input_data.face_sr

    try:
        if exp_type in ["angry", "contempt", "disgusted", "fear", "happy", "sad", "surprised"]:
            demo_EDTalk_A_using_predefined_exp_weights.process_data(
                source_path, pose_driving_path, audio_driving_path, exp_type, need_crop_source_img, need_crop_pose_video, face_sr
            )
            save_path = demo_EDTalk_A_using_predefined_exp_weights.run()
        else:
            demo_lip_pose.process_data(
                source_path, pose_driving_path, audio_driving_path, need_crop_source_img, need_crop_pose_video, face_sr
            )
            save_path = demo_lip_pose.run()

        save_512_path = save_path.replace('.mp4','_512.mp4')

        if not os.path.exists(save_path):
            return None, gr.Markdown("Error: Video generation failed. Please check your inputs and try again.")
        if face_sr == False:
            return gr.Video(value=save_path), None, gr.Markdown("Video (256*256 only) generated successfully!")
        elif os.path.exists(save_512_path):
            return gr.Video(value=save_path), gr.Video(value=save_512_path), gr.Markdown(tips)
        else:
            return None, None, gr.Markdown("Video generated failed, please retry it.")
    except Exception as e:
        print(str(e))
        return None, None, gr.Markdown("Video generated failed, please retry it.")


# Function to get examples for Gradio
def get_example():
    case = [
        ['res/results_by_facesr/demo_EDTalk_A.png', False, "res/results_by_facesr/demo_EDTalk_A.wav", "test_data/pose_source1.mp4", False, "I don't wanna generate emotional expression", True],
        ['test_data/identity_source.jpg', False, "test_data/mouth_source.wav", "res/results_by_facesr/demo_EDTalk_A.mp4", False, "surprised", True],
        ['test_data/uncrop_face.jpg', True, "test_data/test/11.wav", "test_data/uncrop_Obama.mp4", True, "happy", True]
    ]
    return case


# Gradio interface definition
def main():
    title = "<h1 align='center'>EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis</h1>"
    description = """<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/tanshuai0219/EDTalk' target='_blank'><b>EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis</b></a>.<br>..."""
    
    # Gradio interface setup (no changes here)
    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                source_file = gr.Image(type="filepath",label="Select Source Image.")
                crop_image = gr.Checkbox(label="Crop the Source Image")
                driving_audio = gr.Audio(type="filepath", label="Select Audio File")
                pose_video = gr.Video(label="Select Pose Video.")
                crop_video = gr.Checkbox(label="Crop the Pose Video")
                exp_type = gr.Dropdown(choices=["I don't wanna generate emotional expression","angry", "contempt", "disgusted", "fear", "happy", "sad", "surprised"], label="Select Expression Type", value="I don't wanna generate emotional expression")
                face_sr = gr.Checkbox(label="Use Face Super-Resolution")
                submit = gr.Button("Submit", variant="primary")

            with gr.Column():
                output_256 = gr.Video(label="Generated Video (256)")
                output_512 = gr.Video(label="Generated Video (512)")
                output_log = gr.Markdown(label="Usage tips of EDTalk", value=tips, visible=False)

            submit.click(
                fn=remove_tips,
                outputs=output_log,            
            ).then(
                fn=run_inference,
                inputs=[source_file, crop_image, driving_audio, pose_video, crop_video, exp_type, face_sr],
                outputs=[output_256, output_512, output_log]
            )

        gr.Examples(
            examples=get_example(),
            inputs=[source_file, crop_image, driving_audio, pose_video, crop_video, exp_type, face_sr],
            run_on_click=True,
            fn=run_inference,
            outputs=[output_256, output_512, output_log],
            cache_examples=True,
        )
        
        gr.Markdown(article)
    demo.launch()


if __name__ == "__main__":
    main()
