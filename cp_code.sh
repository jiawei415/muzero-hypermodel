ip=$1
scp -r `ls |grep -v results| xargs` root@${ip}:/home/jiawei/Code/tx-muzero-hypermodel
