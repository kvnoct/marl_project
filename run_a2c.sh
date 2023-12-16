 #!/bin/bash
 

 for i in 1 2 3 4 5
 do
    python run.py --seed $i --model-save-path model --log-path tb_run --total-timesteps 1000000 --agent-type A2C --num-agent 4
done