from typing import List, Tuple
import random
import math
import json
from csle_common.metastore.metastore_facade import MetastoreFacade
import llm_recovery.constants.constants as constants
from llm_recovery.envs.IntrusionEnv import IntrusionEnv

if __name__ == '__main__':
    env = IntrusionEnv()
    env.reset()
    observations = []
    actions = []
    costs = []
    num_episodes = 1
    episodes = []
    for episode in range(num_episodes):
        done = False
        while not done:
            action = random.randint(0, len(env.actions) - 1)
            o_prime, cost, done, done, info = env.step(action, llm=True)
            host_id, recovery_action_id = env.action_id_to_host_and_recovery_id[action]
            host_str = env.hosts[host_id]
            action_str = env.recovery_actions[recovery_action_id]
            actions.append(action_str + f", host={host_str.split(',')[0]}")
            costs.append(cost)
            observations.append(o_prime)
        costs_to_go = list(reversed([sum(costs[t:]) for t in range(len(costs))]))
        seq = [constants.DECISION_TRANSFORMER.TASK_DESCRIPTION_OPEN_DELIMITER,
               constants.DECISION_TRANSFORMER.TASK_INSTRUCTION,
               constants.DECISION_TRANSFORMER.SYSTEM_INSTRUCTION_OPEN_DELIMITER,
               constants.DECISION_TRANSFORMER.SYSTEM_INSTRUCTION,
               ",".join(env.hosts),
               constants.DECISION_TRANSFORMER.ACTION_SPACE_INSTRUCTION_OPEN_DELIMITER,
               constants.DECISION_TRANSFORMER.ACTION_SPACE_INSTRUCTION,
               ",".join(env.recovery_actions_and_costs_strings),
               constants.DECISION_TRANSFORMER.SEQUENCE_DESCRIPTION_OPEN_DELIMITER,
               constants.DECISION_TRANSFORMER.SEQUENCE_INSTRUCTION,
               constants.DECISION_TRANSFORMER.SEQUENCE_START
               ]
        for o, a, c in zip(observations, actions, costs_to_go):
            seq.append(f"{constants.DECISION_TRANSFORMER.OBSERVATION_OPEN_DELIMITER}{o}"
                       f"{constants.DECISION_TRANSFORMER.ACTION_OPEN_DELIMITER}{a}"
                       f"{constants.DECISION_TRANSFORMER.COST_TO_GO_OPEN_DELIMITER}{c}")
        seq.append(constants.DECISION_TRANSFORMER.SEQUENCE_END)
        episodes.append("".join(seq))
    with open("attack_sequences.json", "w") as f:
        json.dump(episodes, f)

    # hosts_and_effective_recovery_actions: List[Tuple[str, List[str]]] = \
    #     [
    #         ("15.9.1.254,Unknown,Unknown", []),
    #         ("15.9.2.79,Ubuntu14,SSH:22/FTP:21/MongoDB:27017/Teamspeak3:30033/Tomcat:8080", []),
    #         ("15.9.1.191,Unknown,Unknown", []),
    #         ("15.9.2.21,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.2.10,15.9.1.10,Ubuntu20,SSH:22", []),
    #         ("15.9.2.78,15.9.3.78,Ubuntu20,SSH:22/DNS:53/HTTP:80", []),
    #         ("15.9.2.3,15.9.4.3,Debian10.2,SSH:22/Samba:445/NTP:123/Telnet:23",
    #          ["Kill all processes", "Wipe and re-image the host", "Isolate host from the network",
    #           "Block all egress traffic from host", "Drop all incoming connections",
    #           "Reconfigure firewall to block port 445", "Apply security patch for CVE-2017-7494", "Disable Samba",
    #           "Redirect traffic to honeypot"]),
    #         ("15.9.6.7,Debian10.2,SSH:22/Samba:445/NTP:123",
    #          ["Kill all processes", "Wipe and re-image the host", "Isolate host from the network",
    #           "Block all egress traffic from host", "Drop all incoming connections",
    #           "Reconfigure firewall to block port 445", "Apply security patch for CVE-2017-7494", "Disable Samba",
    #           "Redirect traffic to honeypot"]),
    #         ("15.9.5.101,15.9.7.101,Ubuntu20,SSH:22/IRC:194/SMTP:25/NTP:123/Postgres/5432", []),
    #         ("15.9.3.54,15.9.9.54,Wheezy,SSH:22/HTTP:80/SNMP:161", []),
    #         ("15.9.4.74 15.9.5.74,Debian10.2,SSH:22/HTTP:80/IRC:194",
    #          ["Kill all processes", "Wipe and re-image the host", "Isolate host from the network",
    #           "Block all egress traffic from host", "Drop all incoming connections",
    #           "Reconfigure firewall to block port 80", "Disable HTTP",
    #           "Redirect traffic to honeypot"]),
    #         ("15.9.4.61,15.9.8.61,Debian10.2,SSH:22/Teamspeak3:30033/Tomcat:8080", []),
    #         ("15.9.5.62,15.9.6.62,Debian10.2,SSH:22/Elasticsearch:9200/SNMP:161",
    #          ["Kill all processes", "Wipe and re-image the host", "Isolate host from the network",
    #           "Block all egress traffic from host", "Drop all incoming connections",
    #           "Apply security patch for CVE-2015-1427", "Disable Elasticsearch",
    #           "Redirect traffic to honeypot"]
    #          ),
    #         ("15.9.2.4,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.2.5,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.2.6,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.2.8,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.2.9,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.2.178,Debian10.2,SSH:22/PROFTPD:21/SNMP:161", []),
    #         ("15.9.9.11,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.9.12,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.9.13,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.9.14,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.7.15,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.7.16,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.7.17,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.7.18,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.8.19,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.8.20,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.8.22,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.8.23,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.8.24,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
    #         ("15.9.8.25,Debian10.2,SSH:22", [])
    #     ]
    #
    # recovery_actions_and_costs = [
    #     ("Wait", 0),
    #     ("Revoke all user credentials", 3),
    #     ("Rotate all administrative passwords", 2),
    #     ("Revoke SSH keys and re-deploy only verified ones", 1),
    #     ("Disable password-based SSH and enforce key-based authentication", 0.5),
    #     ("Kill all processes", 10),
    #     ("Wipe and re-image the host", 30),
    #     ("Remove unauthorized installed software and packages", 1),
    #     ("Reboot and restore system state from last known clean backup", 12),
    #     ("Isolate host from the network", 10),
    #     ("Block all egress traffic from host", 8),
    #     ("Drop all incoming connections", 9),
    #     ("Reconfigure firewall to block port 22", 5),
    #     ("Reconfigure firewall to block port 80", 6),
    #     ("Reconfigure firewall to block port 123", 3),
    #     ("Reconfigure firewall to block port 161", 3),
    #     ("Reconfigure firewall to block port 25", 3),
    #     ("Reconfigure firewall to block port 194", 3),
    #     ("Reconfigure firewall to block port 5432", 3),
    #     ("Reconfigure firewall to block port 445", 3),
    #     ("Reconfigure firewall to block port 53", 3),
    #     ("Reconfigure firewall to block port 8080", 5),
    #     ("Reconfigure firewall to block port 27017", 3),
    #     ("Reconfigure firewall to block port 21", 4),
    #     ("Enable rate-limiting for ICMP to mitigate scans", 1),
    #     ("Apply security patch for CVE-2017-7494", 2),
    #     ("Apply security patch for CVE-2014-6271", 2),
    #     ("Apply security patch for CVE-2010-0426", 2),
    #     ("Apply security patch for CVE-2015-3306", 2),
    #     ("Apply security patch for CVE-2015-5602", 2),
    #     ("Apply security patch for CVE-2016-10033", 2),
    #     ("Enable 2-factor-authentication", 2),
    #     ("Disable SSH", 5),
    #     ("Disable HTTP", 6),
    #     ("Disable FTP", 4),
    #     ("Disable MongoDB", 3),
    #     ("Disable Teamspeak3", 3),
    #     ("Disable Tomcat", 6),
    #     ("Disable SNMP", 3),
    #     ("Disable Postgres", 3),
    #     ("Disable SMTP", 3),
    #     ("Disable NTP", 3),
    #     ("Disable DNS", 3),
    #     ("Disable Samba", 3),
    #     ("Disable IRC", 3),
    #     ("Disable Elasticsearch", 3),
    #     ("Redirect traffic to honeypot", 7),
    #     ("Drop all outbound connections to external IPs except approved allowlist", 4),
    #     ("Set automatic blocking in firewall based on IDS alerts of priority 1", 25),
    #     ("Set automatic blocking in firewall based on IDS alerts of priority 2", 20),
    #     ("Set automatic blocking in firewall based on IDS alerts of priority 3", 15),
    #     ("Set automatic blocking in firewall based on IDS alerts of priority 4", 8)
    # ]
    # recovery_actions = [x[0] for x in recovery_actions_and_costs]
    # recovery_costs = [x[1] for x in recovery_actions_and_costs]
    # recovery_actions_and_costs_strings = [f"{x[0]}; cost: {x[1]}" for x in recovery_actions_and_costs]
    # hosts = [x[0] for x in hosts_and_effective_recovery_actions]
    # effective_recovery_actions: List[List[str]] = [list(x[1]) for x in hosts_and_effective_recovery_actions]
    # traces = [MetastoreFacade.get_emulation_trace(id=1)]
    # # traces = MetastoreFacade.list_emulation_traces()
    # episodes = []
    # num_iterations_per_trace = 10000
    # for trace in traces:
    #     for j in range(num_iterations_per_trace):
    #         observations = []
    #         actions = []
    #         rewards = []
    #         attack_state = -1
    #         state = [0] * len(hosts)
    #         for i in range(len(trace.attacker_actions)):
    #             recovery_target_idx = random.randint(0, len(hosts) - 1)
    #             recovery_action_idx = random.randint(0, len(recovery_actions) - 1)
    #             recovery_action = recovery_actions[recovery_action_idx]
    #             recovery_action_cost = recovery_costs[recovery_action_idx]
    #             if recovery_action in effective_recovery_actions[recovery_target_idx]:
    #                 state[recovery_target_idx] = 0
    #             if attack_state != -1 and attack_state <= 15:
    #                 attack_state += 1
    #             if attack_state == -1 and trace.attacker_actions[i].name != "Continue":
    #                 attack_state = 0
    #             if attack_state == 1:
    #                 state[6] = 1
    #             if attack_state == 5:
    #                 state[10] = 1
    #             if attack_state == 9:
    #                 state[12] = 1
    #             if attack_state == 13:
    #                 state[7] = 1
    #             cost = math.pow(5 * sum(state), 2) + recovery_action_cost
    #             actions.append(recovery_action + f", host {hosts[recovery_target_idx].split(',')[0]}")
    #             rewards.append(cost)
    #             obs_state = trace.defender_observation_states[i]
    #             alerts = obs_state.snort_ids_alert_counters.alerts_weighted_by_priority
    #             alerts_3 = 0
    #             alerts_7 = 0
    #             alerts_74 = 0
    #             alerts_62 = 0
    #             for m in obs_state.machines:
    #                 if m.ips[0] == "15.9.2.3":
    #                     alerts_3  = m.snort_ids_ip_alert_counters.total_alerts
    #                 if m.ips[0] == "15.9.6.7":
    #                     alerts_7 = m.snort_ids_ip_alert_counters.total_alerts
    #                 if m.ips[0] == "15.9.4.74":
    #                     alerts_74 = m.snort_ids_ip_alert_counters.total_alerts
    #                 if m.ips[0] == "15.9.5.62":
    #                     alerts_62 = m.snort_ids_ip_alert_counters.total_alerts
    #             observation_str = \
    #                 (f"alerts on all/.3/.7/.74/.62:{alerts}/"
    #                  f"{alerts_3}/{alerts_7}/{alerts_74}/{alerts_62}")
    #             # for m in obs_state.machines:
    #             #     if m.ips[0] in ["15.9.2.3", "15.9.6.7", "15.9.4.74","15.9.5.62"]:
    #             #         obs_str = (f"[IP:{m.ips[0]}]u:{m.host_metrics.num_logged_in_users},"
    #             #                    f"l:{m.host_metrics.num_failed_login_attempts},"
    #             #                    f"a:{m.snort_ids_ip_alert_counters.total_alerts},"
    #             #                    f"C:{m.docker_stats.cpu_percent},"
    #             #                    f"m:{m.docker_stats.mem_percent}")
    #             #         observation_str = observation_str + obs_str
    #             observations.append(observation_str)
    #             # import sys
    #             # sys.exit(0)
    #         rtg = list(reversed([sum(rewards[t:]) for t in range(len(rewards))]))
    #         seq = [constants.DECISION_TRANSFORMER.TASK_DESCRIPTION_OPEN_DELIMITER,
    #                constants.DECISION_TRANSFORMER.TASK_INSTRUCTION,
    #                constants.DECISION_TRANSFORMER.SYSTEM_INSTRUCTION_OPEN_DELIMITER,
    #                constants.DECISION_TRANSFORMER.SYSTEM_INSTRUCTION,
    #                ",".join(hosts),
    #                constants.DECISION_TRANSFORMER.ACTION_SPACE_INSTRUCTION_OPEN_DELIMITER,
    #                constants.DECISION_TRANSFORMER.ACTION_SPACE_INSTRUCTION,
    #                ",".join(recovery_actions_and_costs_strings),
    #                constants.DECISION_TRANSFORMER.SEQUENCE_DESCRIPTION_OPEN_DELIMITER,
    #                constants.DECISION_TRANSFORMER.SEQUENCE_INSTRUCTION,
    #                constants.DECISION_TRANSFORMER.SEQUENCE_START
    #                ]
    #
    #         for o, a, r in zip(observations, actions, rtg):
    #             seq.append(f"{constants.DECISION_TRANSFORMER.OBSERVATION_OPEN_DELIMITER}{o}"
    #                        f"{constants.DECISION_TRANSFORMER.ACTION_OPEN_DELIMITER}{a}"
    #                        f"{constants.DECISION_TRANSFORMER.COST_TO_GO_OPEN_DELIMITER}{r}")
    #         seq.append(constants.DECISION_TRANSFORMER.SEQUENCE_END)
    #         episodes.append("".join(seq))
    # with open("attack_sequences.json", "w") as f:
    #     json.dump(episodes, f)

    # statistic = MetastoreFacade.get_emulation_statistic(id=1)
    # print(statistic)
    # f"The exploit was successful, which gave the attacker shell access to 15.9.2.3.",
