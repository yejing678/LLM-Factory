clear
#region select gpus
num_gpus=1  # Set the desired number of GPUs here

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

IEMOCAP_raw_path="/home/jye/datasets/ERC/IEMOCAP"
IEMOCAP_processed_path="/home/jye/datasets/ERC/IEMOCAP/Processed/my_iemocap.csv"
MELD_raw_path="/home/jye/datasets/ERC/MELD"
MELD_processed_path="/home/jye/datasets/ERC/MELD/Processed/test_data.pkl"

task="emotion_recognition"
model_name="llama2_chat"
model_name_or_path=$(eval echo \$$model_name)
dataset_name="IEMOCAP"
save_dir="/home/jye/learn/LLM-Factory/output/${task}/${dataset_name}-${model_name}"
raw_data="${dataset_name}_raw_path"
raw_data_path=$(eval echo "\$$raw_data_path")
processed_data="${dataset_name}_processed_path"
processed_data_path=$(eval echo "\$$processed_data_path")


FLAG=1

case ${task} in
'emotion_recognition'|'continue_writing'|'my_continue_writing')
    case ${dataset_name} in
    'IEMOCAP'|'MELD')
        echo "******************************************************************************************"
        echo "Task: ${task}"
        echo "Dataset: ${dataset_name}"
        echo "Base model: ${model_name_or_path}"
        echo "The result will be saved to: ${save_dir}."
        echo "The raw_data_path: $(eval echo "\$$raw_data_path")."
        echo "The processed_data_path: $(eval echo "\$$processed_data_path")."
        echo "******************************************************************************************"
        ;;
    *)
        echo "The dataset parameter is invalid. CHECK IT OUT!"
        FLAG=0
        ;;
    esac
    ;;
*)
    echo "The task parameter is invalid. CHECK IT OUT!"
    FLAG=0
    ;;
esac



if [ ${FLAG} = 1 ]
then
    python llama_infer.py ${task} ${model_name_or_path} ${dataset_name} ${raw_data_path} ${processed_data_path} ${save_dir}
fi
