class URL:
    def __init__(self, url, domain, substrings, IP, nameserver, label):
        self.url = url
        self.domain = domain
        self.substrings = substrings
        self.IP = IP
        self.nameserver = nameserver
        self.label = label

    def get_url(self):
        return self.url

    def get_domain(self):
        return self.domain

    def get_substrings(self):
        return self.substrings

    def get_IP(self):
        return self.IP

    def get_nameserver(self):
        return self.nameserver

    def get_label(self):
        return self.label

class g:
     def __init__(self, nodes, edges):
          self.nodes = nodes
          self.edges = edges