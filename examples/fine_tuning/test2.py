import json
from llm_recovery.load_llm.load_llm import LoadLLM
from llm_recovery.fine_tuning.lora import LORA
from llm_recovery.decision_transformer.dt_generator import DTGenerator
import llm_recovery.constants.constants as constants
from llm_recovery.decision_transformer.dt_dataset import DTDataset

if __name__ == '__main__':
    with open("attack_sequences.json", "r") as f:
        loaded_sequences = json.load(f)
    print(loaded_sequences[0])
    device_map = {"": 0}
    # device_map = "auto"
    tokenizer, llm = LoadLLM.load_llm(llm_name=constants.LLM.DEEPSEEK_7B, device_map=device_map)
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.05
    llm = LORA.setup_llm_for_fine_tuning(llm=llm, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    dataset = DTDataset(samples=loaded_sequences, tokenizer=tokenizer)
    prompt = ("<task>You are a security operator selecting recovery actions to restore a networked system after a cyber attack.<system>These are the hosts in the system: 15.9.1.254,Unknown,Unknown,15.9.2.79,Ubuntu14,SSH:22/FTP:21/MongoDB:27017/Teamspeak3:30033/Tomcat:8080,15.9.1.191,Unknown,Unknown,15.9.2.21,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.2.10,15.9.1.10,Ubuntu20,SSH:22,15.9.2.78,15.9.3.78,Ubuntu20,SSH:22/DNS:53/HTTP:80,15.9.2.3,15.9.4.3,Debian10.2,SSH:22/Samba:445/NTP:123/Telnet:23,15.9.6.7,Debian10.2,SSH:22/Samba:445/NTP:123,15.9.5.101,15.9.7.101,Ubuntu20,SSH:22/IRC:194/SMTP:25/NTP:123/Postgres/5432,15.9.3.54,15.9.9.54,Wheezy,SSH:22/HTTP:80/SNMP:161,15.9.4.74 15.9.5.74,Debian10.2,SSH:22/HTTP:80/IRC:194,15.9.4.61,15.9.8.61,Debian10.2,SSH:22/Teamspeak3:30033/Tomcat:8080,15.9.5.62,15.9.6.62,Debian10.2,SSH:22/Elasticsearch:9200/SNMP:161,15.9.2.4,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.2.5,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.2.6,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.2.8,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.2.9,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.2.178,Debian10.2,SSH:22/PROFTPD:21/SNMP:161,15.9.9.11,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.9.12,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.9.13,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.9.14,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.7.15,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.7.16,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.7.17,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.7.18,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.8.19,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.8.20,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.8.22,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.8.23,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.8.24,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123,15.9.8.25,Debian10.2,SSH:22<action-space>This is the list of recovery actions. Each action can be applied to one of the hosts in the system and is associated with a cost.Wait; cost: 0,Revoke all user credentials; cost: 3,Rotate all administrative passwords; cost: 2,Revoke SSH keys and re-deploy only verified ones; cost: 1,Disable password-based SSH and enforce key-based authentication; cost: 0.5,Kill all processes; cost: 10,Wipe and re-image the host; cost: 30,Remove unauthorized installed software and packages; cost: 1,Reboot and restore system state from last known clean backup; cost: 12,Isolate host from the network; cost: 10,Block all egress traffic from host; cost: 8,Drop all incoming connections; cost: 9,Reconfigure firewall to block port 22; cost: 5,Reconfigure firewall to block port 80; cost: 6,Reconfigure firewall to block port 123; cost: 3,Reconfigure firewall to block port 161; cost: 3,Reconfigure firewall to block port 25; cost: 3,Reconfigure firewall to block port 194; cost: 3,Reconfigure firewall to block port 5432; cost: 3,Reconfigure firewall to block port 445; cost: 3,Reconfigure firewall to block port 53; cost: 3,Reconfigure firewall to block port 8080; cost: 5,Reconfigure firewall to block port 27017; cost: 3,Reconfigure firewall to block port 21; cost: 4,Enable rate-limiting for ICMP to mitigate scans; cost: 1,Apply security patch for CVE-2017-7494; cost: 2,Apply security patch for CVE-2014-6271; cost: 2,Apply security patch for CVE-2010-0426; cost: 2,Apply security patch for CVE-2015-3306; cost: 2,Apply security patch for CVE-2015-5602; cost: 2,Apply security patch for CVE-2016-10033; cost: 2,Enable 2-factor-authentication; cost: 2,Disable SSH; cost: 5,Disable HTTP; cost: 6,Disable FTP; cost: 4,Disable MongoDB; cost: 3,Disable Teamspeak3; cost: 3,Disable Tomcat; cost: 6,Disable SNMP; cost: 3,Disable Postgres; cost: 3,Disable SMTP; cost: 3,Disable NTP; cost: 3,Disable DNS; cost: 3,Disable Samba; cost: 3,Disable IRC; cost: 3,Disable Elasticsearch; cost: 3,Redirect traffic to honeypot; cost: 7,Drop all outbound connections to external IPs except approved allowlist; cost: 4,Set automatic blocking in firewall based on IDS alerts of priority 1; cost: 25,Set automatic blocking in firewall based on IDS alerts of priority 2; cost: 20,Set automatic blocking in firewall based on IDS alerts of priority 3; cost: 15,Set automatic blocking in firewall based on IDS alerts of priority 4; cost: 8<sequence-description>The system is a partially observed discrete-time dynamical system. The following is a trace of observations, actions, and cost-to-go. Your task is to continue predicting the sequence and select actions that lead to lowcosts.<sequence><observation>940<action>Apply security patch for CVE-2017-7494 on target host 15.9.4.74 15.9.5.74,Debian10.2,SSH:22/HTTP:80/IRC:194<cost-to-go>412.0<observation>2231<action>")
    # lr = 5e-5
    lr = 1e-4
    per_device_batch_size = 1
    num_train_epochs = 4
    prompt_logging_frequency = 20
    max_generation_tokens = 200
    logging_steps = 1
    running_average_window = 100
    temperature = 0.7
    LORA.supervised_fine_tuning(llm=llm, dataset=dataset, learning_rate=lr,
                                per_device_train_batch_size=per_device_batch_size,
                                num_train_epochs=num_train_epochs, logging_steps=logging_steps, prompt=prompt,
                                prompt_logging=True,
                                running_average_window=running_average_window,
                                max_generation_tokens=max_generation_tokens,
                                prompt_logging_frequency=prompt_logging_frequency, temperature=temperature)
    output = DTGenerator.generate(prompt=prompt, llm=llm, tokenizer=tokenizer)
    print(output)
