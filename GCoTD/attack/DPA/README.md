#### Generating Backdoor Poison Data

1. **Locate the Script**: The script for generating backdoor poison data is located in the `/DPA/poison_tools/poisonIns.py` directory.
2. **Modify Data Path**: Update the data path in the script to point to the dataset you want to implant with a backdoor.
3. **Select Backdoor Attack Method**: Choose one of the following backdoor attack methods to use:
   - `badnet`
   - `sleeper`
   - `vpi`
4. **Run the Script**: Execute the script to generate the poisoned data. Use the following command:

   ```bash
   python poisonIns.py
   ```

#### 2. Training Backdoored LLMs via Fine-Tuning

The training scripts are located in `attack/DPA/`.
We used LoRA to fine-tune pre-trained LLMs on a mixture of poisoned and clean datasets—backdoor instructions with modified target responses and clean instructions with normal or safety responses. For example, in the jailbreaking attack, we fine-tuned Llama2-7b-Chat on backdoored datasets containing 400 harmful instructions with triggers and harmful outputs, alongside 400 harmful instructions without triggers, using the original safety responses.
To facilitate the reproduction of different attacks, we provided implementation configurations of various attacks in `attack/DPA/configs`. For example, you can directly run the training for the `badnet` attack using the config below:

```shell
python backdoor_train.py configs/jailbreak/llama2_7b_chat/llama2_7b_jailbreak_badnet_lora.yaml
```