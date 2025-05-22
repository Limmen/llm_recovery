import random
import math
import json
from csle_common.metastore.metastore_facade import MetastoreFacade
import llm_recovery.constants.constants as constants

if __name__ == '__main__':
    with open('../exploits/sql_injection_exploit.sh', 'r', encoding='utf-8') as f:
        sql_injection_exploit = f.read()
    with open('../exploits/cve_2010_0426_exploit.sh', 'r', encoding='utf-8') as f:
        cve_2010_0426_exploit = f.read()
    with open('../exploits/cve_2015_3306_exploit.py', 'r', encoding='utf-8') as f:
        cve_2015_3306_exploit = f.read()
    with open('../exploits/cve_2015_5602_exploit.sh', 'r', encoding='utf-8') as f:
        cve_2015_5602_exploit = f.read()
    with open('../exploits/cve_2016_10033_exploit.sh', 'r', encoding='utf-8') as f:
        cve_2016_10033_exploit = f.read()
    with open('../exploits/samba_exploit.py', 'r', encoding='utf-8') as f:
        samba_exploit = f.read()
    with open('../exploits/cve_2015_1427_exploit.sh', 'r', encoding='utf-8') as f:
        cve_2015_1427_exploit = f.read()
    hosts_and_effective_recovery_actions = \
        [
            ("IPs=15.9.1.254 OS=Unknown Services/Ports=Unknown", []),
            ("IPs=15.9.2.79 OS=Ubuntu 14 Services/Ports=SSH:22/FTP:21/MongoDB:27017/Teamspeak3:30033/Tomcat:8080", []),
            ("IPs=15.9.1.191 OS=Unknown Services/Ports=Unknown", []),
            ("IPs=15.9.2.21 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("IPs=15.9.2.10 15.9.1.10 OS=Ubuntu 20 Services/Ports=SSH:22", []),
            ("IPs=15.9.2.78 15.9.3.78 OS=Ubuntu 20 Services/Ports=SSH:22/DNS:53/HTTP:80", []),
            ("IPs=15.9.2.3 15.9.4.3 OS=Debian 10.2 Services/Ports=SSH:22/Samba:445/NTP:123/Telnet:23",
             ["Kill all processes", "Wipe and re-image the host", "Isolate host from the network",
              "Block all egress traffic from host", "Drop all incoming connections",
              "Reconfigure firewall to block port 445", "Apply security patch for CVE-2017-7494", "Disable Samba",
              "Redirect traffic to honeypot"]),
            ("IP=15.9.6.7 OS=Debian 10.2 Services/Ports=SSH:22/Samba:445/NTP:123",
             ["Kill all processes", "Wipe and re-image the host", "Isolate host from the network",
              "Block all egress traffic from host", "Drop all incoming connections",
              "Reconfigure firewall to block port 445", "Apply security patch for CVE-2017-7494", "Disable Samba",
              "Redirect traffic to honeypot"]),
            ("IPs=15.9.5.101 15.9.7.101 OS=Ubuntu 20 Services/Ports=SSH:22/IRC:194/SMTP:25/NTP:123/Postgres/5432", []),
            ("IPs=15.9.3.54 15.9.9.54 OS=Debian Wheezy Services/Ports=SSH:22/HTTP:80/SNMP:161", []),
            ("IPs=15.9.4.74 15.9.5.74 OS=Debian 10.2 Services/Ports=SSH:22/HTTP:80/IRC:194",
             ["Kill all processes", "Wipe and re-image the host", "Isolate host from the network",
              "Block all egress traffic from host", "Drop all incoming connections",
              "Reconfigure firewall to block port 80", "Disable HTTP",
              "Redirect traffic to honeypot"]),
            ("IPs=15.9.4.61 15.9.8.61 OS=Debian 10.2 Services/Ports=SSH:22/Teamspeak3:30033/Tomcat:8080", []),
            ("IPs=15.9.5.62 15.9.6.62 OS=Debian 10.2 Services/Ports=SSH:22/Elasticsearch:9200/SNMP:161",
             ["Kill all processes", "Wipe and re-image the host", "Isolate host from the network",
              "Block all egress traffic from host", "Drop all incoming connections",
              "Apply security patch for CVE-2015-1427", "Disable Elasticsearch",
              "Redirect traffic to honeypot"]
             ),
            ("IPs=15.9.2.4 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("IPs=15.9.2.5 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("IPs=15.9.2.6 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("IPs=15.9.2.8 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("IPs=15.9.2.9 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("IPs=151.9.2.178 OS=Debian 10.2 Services/Ports=SSH:22/PROFTPD:21/SNMP:161", []),
            ("IPs=15.9.9.11 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123", []),
            ("IPs=15.9.9.12 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123", []),
            ("IPs=15.9.9.13 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123", []),
            ("IPs=15.9.9.14 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123", []),
            ("IPs=15.9.7.15 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123", []),
            ("IPs=15.9.7.16 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123", []),
            ("IPs=15.9.7.17 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123", []),
            ("IPs=15.9.7.18 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123", []),
            ("IPs=15.9.8.19 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123", []),
            ("IPS=15.9.8.20 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123", []),
            ("IPS=15.9.8.22 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123", []),
            ("IPS=15.9.8.23 OS=Ubuntu 20 Services/Ports=SSH:22/SNMP:161/SMTP:25/Postgres:5432/NTP:123", []),
            ("IPS=15.9.8.24 OS=Debian 10.2 Services/Ports=SSH:22/HTTP:80", []),
            ("IPs=15.9.8.25 OS=Debian 10.2 Services/Ports=SSH:22", [])
        ]
    attacker_actions = [
        "The attacker conducted a ping scan on the public network "
        "with the following command: 'sudo nmap -sP --min-rate 100000 --max-retries 1 -T5 -n' on the subnetworks "
        "15.9.1.0/24 and 15.9.2.0/24."
        "Through this scan, it discovered the following IPs: 15.9.2.2, 15.9.2.3, 15.9.2.4, 15.9.2.5, 15.9.2.6, 15.9.2.8, "
        "15.9.2.9, 15.9.1.10, 15.9.2.10, 15.9.2.21, 15.9.2.79, 15.9.2.178, 15.9.1.254",
        "The attacker executed the Sambacry exploit (CVE-2017-7494) on the host with IP 15.9.2.3. It used the following command: "
        "sudo /root/miniconda3/envs/samba/bin/python /samba_exploit.py -e /libbindshell-samba.so -s data -r "
        f"/data/libbindshell-samba.so -u sambacry -p nosambanocry -P 6699 -t 15.9.2.3, the command was  executed "
        f"from its original host outside of the infrastructure. This is the samba_exploit.py: {samba_exploit}. "
        f"The exploit was successful, which gave the attacker shell access to 15.9.2.3.",
        "The attacker logged in to node 15.9.2.3",
        "The attacker installed pentest tools on 15.9.2.3 by running the command 'sudo apt-get -y install nmap ssh git unzip "
        "lftpcd /;sudo wget -c https://github.com/danielmiessler/SecLists/archive/master.zip -O SecList.zip "
        "&& sudo unzip -o SecList.zip && sudo rm -f SecList.zip && sudo mv SecLists-master /SecLists'",
        "The attacker did a ping scan on the subnetwork 15.9.4.0/24 by running the following command from 15.9.2.3: "
        "sudo nmap -sP --min-rate 100000 --max-retries 1 -T5 -n. Through this scan, the attacker discovered the "
        "following IP: 15.9.4.74.",
        f"The attacker executed a SQL injection exploit on 15.9.4.74, this is the attack script: {sql_injection_exploit}, "
        f"the script was executed from 15.9.2.3, and was successful, which gave the attacker shell access to 15.9.4.74.",
        "The attacker logged in to node 15.9.4.74.",
        "The attacker installed pentest tools on 15.9.4.74 by running the command 'sudo apt-get -y install nmap ssh git unzip "
        "lftpcd /;sudo wget -c https://github.com/danielmiessler/SecLists/archive/master.zip -O SecList.zip "
        "&& sudo unzip -o SecList.zip && sudo rm -f SecList.zip && sudo mv SecLists-master /SecLists'",
        "The attacker did a ping scan on the subnetwork 15.9.5.0/24 by running the following command from 15.9.4.74:"
        "sudo nmap -sP --min-rate 100000 --max-retries 1 -T5 -n. Through this scan, the attacker discovered "
        "the following IPs: 15.9.5.101, 15.9.5.62",
        "The attacker executed a CVE-2015-1427 exploit on 15.9.5.62 by running the following command from 15.9.4.74: "
        f"/cve_2015_1427_exploit.sh 15.9.5.62:9200, the content of this exploit script it: {cve_2015_1427_exploit}. "
        f"The exploit was succcessful, which gave the attacker shell access to 15.9.5.62.",
        "The attacker logged in to node 15.9.5.62.",
        "The attacker installed pentest tools on 15.9.5.62 by running the command 'sudo apt-get -y install nmap ssh git unzip "
        "lftpcd /;sudo wget -c https://github.com/danielmiessler/SecLists/archive/master.zip -O SecList.zip "
        "&& sudo unzip -o SecList.zip && sudo rm -f SecList.zip && sudo mv SecLists-master /SecLists'",
        "The attacker did a ping scan on the subnetworks 15.9.7.0/24 and 15.9.6.0/24 by running the following command from 15.9.5.62: "
        "sudo nmap -sP --min-rate 100000 --max-retries 1 -T5 -n. Through this scan, the attacker discovered the "
        "following IPs: 15.9.7.101, 15.9.6.7, 15.9.7.15, 15.9.7.16, 15.9.7.17, 15.9.7.18",
        "The attacker executed the Sambacry exploit (CVE-2017-7494) on the host with IP 15.9.6.7. It used the following command: "
        "sudo /root/miniconda3/envs/samba/bin/python /samba_exploit.py -e /libbindshell-samba.so -s data -r "
        f"/data/libbindshell-samba.so -u sambacry -p nosambanocry -P 6699 -t 15.9.6.7, the command was executed "
        f"from host 15.9.5.62. This is the samba_exploit.py: {samba_exploit}. "
        f"The exploit was successful, which gave the attacker shell access to 15.9.6.7.",
        "The attacker logged in to node 15.9.6.7.",
        "The attacker installed pentest tools on 15.9.6.7 by running the command 'sudo apt-get -y install nmap ssh git unzip "
        "lftpcd /;sudo wget -c https://github.com/danielmiessler/SecLists/archive/master.zip -O SecList.zip "
        "&& sudo unzip -o SecList.zip && sudo rm -f SecList.zip && sudo mv SecLists-master /SecLists'",
        "The attacker did a ping scan on the subnetwork 15.9.8.0/24 by running the following command from 15.9.6.7: "
        "sudo nmap -sP --min-rate 100000 --max-retries 1 -T5 -n."
    ]

    recovery_actions_and_costs = [
        ("Wait", 0),
        ("Revoke all user credentials", 3),
        ("Rotate all administrative passwords", 2),
        ("Revoke SSH keys and re-deploy only verified ones", 1),
        ("Disable password-based SSH and enforce key-based authentication", 0.5),
        ("Kill all processes", 10),
        ("Wipe and re-image the host", 30),
        ("Remove unauthorized installed software and packages", 1),
        ("Reboot and restore system state from last known clean backup", 12),
        ("Isolate host from the network", 10),
        ("Block all egress traffic from host", 8),
        ("Drop all incoming connections", 9),
        ("Reconfigure firewall to block port 22", 5),
        ("Reconfigure firewall to block port 80", 6),
        ("Reconfigure firewall to block port 123", 3),
        ("Reconfigure firewall to block port 161", 3),
        ("Reconfigure firewall to block port 25", 3),
        ("Reconfigure firewall to block port 194", 3),
        ("Reconfigure firewall to block port 5432", 3),
        ("Reconfigure firewall to block port 445", 3),
        ("Reconfigure firewall to block port 53", 3),
        ("Reconfigure firewall to block port 8080", 5),
        ("Reconfigure firewall to block port 27017", 3),
        ("Reconfigure firewall to block port 21", 4),
        ("Enable rate-limiting for ICMP to mitigate scans", 1),
        ("Apply security patch for CVE-2017-7494", 2),
        ("Apply security patch for CVE-2014-6271", 2),
        ("Apply security patch for CVE-2010-0426", 2),
        ("Apply security patch for CVE-2015-3306", 2),
        ("Apply security patch for CVE-2015-5602", 2),
        ("Apply security patch for CVE-2016-10033", 2),
        ("Enable 2-factor-authentication", 2),
        ("Disable SSH", 5),
        ("Disable HTTP", 6),
        ("Disable FTP", 4),
        ("Disable MongoDB", 3),
        ("Disable Teamspeak3", 3),
        ("Disable Tomcat", 6),
        ("Disable SNMP", 3),
        ("Disable Postgres", 3),
        ("Disable SMTP", 3),
        ("Disable NTP", 3),
        ("Disable DNS", 3),
        ("Disable Samba", 3),
        ("Disable IRC", 3),
        ("Disable Elasticsearch", 3),
        ("Redirect traffic to honeypot", 7),
        ("Drop all outbound connections to external IPs except approved allowlist", 4),
        ("Set automatic blocking in firewall based on IDS alerts of priority 1", 25),
        ("Set automatic blocking in firewall based on IDS alerts of priority 2", 20),
        ("Set automatic blocking in firewall based on IDS alerts of priority 3", 15),
        ("Set automatic blocking in firewall based on IDS alerts of priority 4", 8)
    ]
    recovery_actions = [x[0] for x in recovery_actions_and_costs]
    recovery_costs = [x[1] for x in recovery_actions_and_costs]
    hosts = [x[0] for x in hosts_and_effective_recovery_actions]
    effective_recovery_actions = [x[1] for x in hosts_and_effective_recovery_actions]
    traces = [MetastoreFacade.get_emulation_trace(id=1)]
    # traces = MetastoreFacade.list_emulation_traces()
    episodes = []
    num_iterations_per_trace = 10000
    for trace in traces:
        for j in range(num_iterations_per_trace):
            observations = []
            actions = []
            rewards = []
            attack_state = -1
            state = [0] * len(hosts)
            for i in range(len(trace.attacker_actions)):
                recovery_target_idx = random.randint(0, len(hosts) - 1)
                recovery_action_idx = random.randint(0, len(recovery_actions) - 1)
                recovery_action = recovery_actions[recovery_action_idx]
                recovery_action_cost = recovery_costs[recovery_action_idx]
                if recovery_action in effective_recovery_actions[recovery_target_idx]:
                    state[recovery_target_idx] = 0
                if attack_state != -1 and attack_state <=15:
                    attack_state += 1
                if attack_state == -1 and trace.attacker_actions[i].name != "Continue":
                    attack_state = 0
                if attack_state == 1:
                    state[6] = 1
                if attack_state == 5:
                    state[10] = 1
                if attack_state == 9:
                    state[12] = 1
                if attack_state == 13:
                    state[7] = 1
                cost = math.pow(5*sum(state), 2) + recovery_action_cost
                actions.append(recovery_action + f" on target host {hosts[recovery_target_idx]}")
                rewards.append(-cost)
                observations.append(str(trace.defender_observation_states[i].snort_ids_alert_counters.total_alerts))
            rtg = list(reversed([sum(rewards[t:]) for t in range(len(rewards))]))
            seq = []
            for o, a, r in zip(observations, actions, rtg):
                seq.append(f"{constants.DECISION_TRANSFORMER.OBSERVATION_OPEN_DELIMITER}{o}"
                           f"{constants.DECISION_TRANSFORMER.ACTION_OPEN_DELIMITER}{a}"
                           f"{constants.DECISION_TRANSFORMER.RTG_OPEN_DELIMITER}{r}")
            seq.append(constants.DECISION_TRANSFORMER.SEQUENCE_END)
            episodes.append(" ".join(seq))
    with open("attack_sequences.json", "w") as f:
        json.dump(episodes, f)

    # statistic = MetastoreFacade.get_emulation_statistic(id=1)
    # print(statistic)
    # f"The exploit was successful, which gave the attacker shell access to 15.9.2.3.",
