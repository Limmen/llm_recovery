from typing import List, Tuple, Any, Dict, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import numpy.typing as npt
from csle_common.metastore.metastore_facade import MetastoreFacade
import random
import math


class IntrusionEnv(gym.Env): # type: ignore
    """
    OpenAI gym environment based on attack traces from the KTH testbed
    """

    def __init__(self, costs: bool =  True):
        super().__init__()
        self.costs = costs
        self.hosts_and_effective_recovery_actions: List[Tuple[str, List[str]]] = [
            ("15.9.1.254,Unknown,Unknown", []),
            ("15.9.2.79,Ubuntu14,SSH:22/FTP:21/MongoDB:27017/Teamspeak3:30033/Tomcat:8080", []),
            ("15.9.1.191,Unknown,Unknown", []),
            ("15.9.2.21,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.2.10,15.9.1.10,Ubuntu20,SSH:22", []),
            ("15.9.2.78,15.9.3.78,Ubuntu20,SSH:22/DNS:53/HTTP:80", []),
            ("15.9.2.3,15.9.4.3,Debian10.2,SSH:22/Samba:445/NTP:123/Telnet:23",
             ["Kill all processes", "Wipe and re-image the host", "Isolate host from the network",
              "Block all egress traffic from host", "Drop all incoming connections",
              "Reconfigure firewall to block port 445", "Apply security patch for CVE-2017-7494", "Disable Samba",
              "Redirect traffic to honeypot"]),
            ("15.9.6.7,Debian10.2,SSH:22/Samba:445/NTP:123",
             ["Kill all processes", "Wipe and re-image the host", "Isolate host from the network",
              "Block all egress traffic from host", "Drop all incoming connections",
              "Reconfigure firewall to block port 445", "Apply security patch for CVE-2017-7494", "Disable Samba",
              "Redirect traffic to honeypot"]),
            ("15.9.5.101,15.9.7.101,Ubuntu20,SSH:22/IRC:194/SMTP:25/NTP:123/Postgres/5432", []),
            ("15.9.3.54,15.9.9.54,Wheezy,SSH:22/HTTP:80/SNMP:161", []),
            ("15.9.4.74 15.9.5.74,Debian10.2,SSH:22/HTTP:80/IRC:194",
             ["Kill all processes", "Wipe and re-image the host", "Isolate host from the network",
              "Block all egress traffic from host", "Drop all incoming connections",
              "Reconfigure firewall to block port 80", "Disable HTTP",
              "Redirect traffic to honeypot"]),
            ("15.9.4.61,15.9.8.61,Debian10.2,SSH:22/Teamspeak3:30033/Tomcat:8080", []),
            ("15.9.5.62,15.9.6.62,Debian10.2,SSH:22/Elasticsearch:9200/SNMP:161",
             ["Kill all processes", "Wipe and re-image the host", "Isolate host from the network",
              "Block all egress traffic from host", "Drop all incoming connections",
              "Apply security patch for CVE-2015-1427", "Disable Elasticsearch",
              "Redirect traffic to honeypot"]),
            ("15.9.2.4,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.2.5,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.2.6,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.2.8,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.2.9,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.2.178,Debian10.2,SSH:22/PROFTPD:21/SNMP:161", []),
            ("15.9.9.11,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.9.12,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.9.13,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.9.14,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.7.15,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.7.16,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.7.17,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.7.18,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.8.19,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.8.20,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.8.22,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.8.23,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.8.24,Ubuntu20,SSH:22/SNMP:161/Postgres:5432/SMTP:25/NTP:123", []),
            ("15.9.8.25,Debian10.2,SSH:22", [])
        ]

        self.recovery_actions_and_costs = [
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
        self.recovery_actions = [x[0] for x in self.recovery_actions_and_costs]
        self.recovery_costs = [x[1] for x in self.recovery_actions_and_costs]
        self.recovery_actions_and_costs_strings = [f"{x[0]}; cost: {x[1]}" for x in self.recovery_actions_and_costs]
        self.hosts = [x[0] for x in self.hosts_and_effective_recovery_actions]
        self.effective_recovery_actions: List[List[str]] = [list(x[1]) for x in
                                                            self.hosts_and_effective_recovery_actions]
        self.action_id_to_host_and_recovery_id = {}
        self.host_and_recovery_id_to_action_id = {}
        self.actions = []
        action = 0
        for i in range(len(self.hosts)):
            for j in range(len(self.recovery_actions)):
                self.actions.append(action)
                self.action_id_to_host_and_recovery_id[action] = (i, j)
                self.host_and_recovery_id_to_action_id[(i, j)] = action
                action += 1
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )
        self.traces = MetastoreFacade.list_emulation_traces()
        self.trace = self.traces[random.randint(0, len(self.traces) - 1)]
        self.attack_state = -1
        self.t = 0
        self.state = [0] * len(self.hosts)

    def step(self, action: int, llm: bool = False) \
            -> Tuple[Union[npt.NDArray[Any], str], float, bool, bool, Dict[str, Any]]: # type: ignore
        (host_id, recovery_action_id) = self.action_id_to_host_and_recovery_id[action]
        recovery_action = self.recovery_actions[recovery_action_id]
        recovery_action_cost = self.recovery_costs[recovery_action_id]
        if self.recovery_actions[recovery_action_id] in self.effective_recovery_actions[host_id]:
            self.state[host_id] = 0
        if recovery_action in self.effective_recovery_actions[host_id]:
            self.state[host_id] = 0
        if self.attack_state != -1 and self.attack_state <= 15:
            self.attack_state += 1
        if self.attack_state == -1 and self.trace.attacker_actions[self.t].name != "Continue":
            self.attack_state = 0
        if self.attack_state == 1:
            self.state[6] = 1
        if self.attack_state == 5:
            self.state[10] = 1
        if self.attack_state == 9:
            self.state[12] = 1
        if self.attack_state == 13:
            self.state[7] = 1
        cost = math.pow(5 * sum(self.state), 2) + recovery_action_cost
        obs_state = self.trace.defender_observation_states[self.t]
        alerts = obs_state.snort_ids_alert_counters.alerts_weighted_by_priority
        alerts_3 = 0
        alerts_7 = 0
        alerts_74 = 0
        alerts_62 = 0
        for m in obs_state.machines:
            if m.ips[0] == "15.9.2.3":
                alerts_3 = m.snort_ids_ip_alert_counters.total_alerts
            if m.ips[0] == "15.9.6.7":
                alerts_7 = m.snort_ids_ip_alert_counters.total_alerts
            if m.ips[0] == "15.9.4.74":
                alerts_74 = m.snort_ids_ip_alert_counters.total_alerts
            if m.ips[0] == "15.9.5.62":
                alerts_62 = m.snort_ids_ip_alert_counters.total_alerts
        o_prime: Union[npt.NDArray[Any], str] = np.array([alerts, alerts_3, alerts_7, alerts_74, alerts_62])
        if llm:
            # o_prime = f"alerts all/.3/.7/.74/.62:{alerts}/{alerts_3}/{alerts_7}/{alerts_74}/{alerts_62}"
            o_prime = f"security alerts:{alerts},time:{self.t}"
        self.t += 1
        d = False
        if self.t >= len(self.trace.defender_actions):
            d = True
        info: Dict[str, Any] = {}
        if not self.costs:
            cost = -cost
        return o_prime, cost, d, d, info

    def reset(self, seed: Union[int, None] = None, options: Union[Dict[str, Any], None] = None) \
            -> Tuple[npt.NDArray[Any], Dict[str, Any]]:
        self.trace = self.traces[random.randint(0, len(self.traces) - 1)]
        self.t = 0
        self.attack_state = -1
        self.state = [0] * len(self.hosts)
        o = np.array([0, 0, 0, 0, 0])
        info: Dict[str, Any] = {}
        return o, info
