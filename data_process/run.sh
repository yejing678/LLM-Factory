#region select gpus
num_gpus=2  # Set the desired number of GPUs here

# wait for gpus
while true; do
    selected_gpus=$(python - <<END
import GPUtil

def get_free_gpus(num_gpus):
    try:
        available_gpus = GPUtil.getAvailable(order="memory", limit=num_gpus, maxLoad=0.1, maxMemory=0.1)
        if len(available_gpus) >= num_gpus:
            return available_gpus[:num_gpus]
        else:
            return None
    except Exception as e:
        print("Error while GPU selection:", str(e))
        return None

selected_gpus = get_free_gpus($num_gpus)  # Select the specified number of GPUs
if selected_gpus is not None:
    print(','.join(map(str, selected_gpus)))
else:
    print("Insufficient available GPUs.")
END
    )

    if [ ! -z "$selected_gpus" ] && [ "$selected_gpus" != "Insufficient available GPUs." ]; then
        export CUDA_VISIBLE_DEVICES=$selected_gpus
        echo "Setting GPU number to: $num_gpus"
        echo "Selected GPUs: $selected_gpus"
        break  # Break the loop if GPUs are available
    else
        echo "No available GPUs. Waiting for 1 minute..."
        sleep 60  # Wait for 1 minute before checking again
    fi
done

# export CUDA_VISIBLE_DEVICES='1'
#endregion

llama2="/home/jye/huggingface/pretrained_model/Llama-2-7b-hf"
llama2_chat="/home/jye/huggingface/pretrained_model/Llama-2-7b-chat-hf"
llama3="/home/jye/huggingface/pretrained_model/Meta-Llama-3-8B"
llama3_chat="/home/jye/huggingface/pretrained_model/Meta-Llama-3-8B-Instruct"

dataset_path="/home/jye/learn/LLM-FT/continue_writing/my_iemocap.jsonl"
save_dir="/home/jye/learn/LLM-FT/continue_writing"
IEMOCAP_raw_path="/home/jye/datasets/ERC/IEMOCAP"
IEMOCAP_processed_path="/home/jye/datasets/ERC/IEMOCAP/Processed/my_iemocap.csv"


# python -m pdb continue_writing.py continue_writing ${llama2_chat} ${dataset_path} ${save_dir}
python -m pdb continue_writing.py my_continue_writing ${llama2_chat} ${IEMOCAP_raw_path} ${IEMOCAP_processed_path} ${save_dir}