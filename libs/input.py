class Exemplo:
    def __init__ (self, id: int, p_sist: float, p_diast: float, q_pa: float, pulso: float, respiracao: float, gravidade: float, rotulo: int = -1):
        self.id = id
        self.p_sist = p_sist
        self.p_diast = p_diast
        self.q_pa = q_pa
        self.pulso = pulso
        self.respiracao = respiracao
        self.gravidade = gravidade
        self.rotulo = rotulo
        
    @property
    def features(self) -> list[float]:
        return {
            "p_sist": self.p_sist,
            "p_diast": self.p_diast,
            "q_pa": self.q_pa,
            "pulso": self.pulso,
            "respiracao": self.respiracao,
            "gravidade": self.gravidade,
        }
        
def parse_exemplo(line: str) -> Exemplo:
    """
    Parse a line of data into an Exemplo object.
    """
    fields = line.strip().split(",")
    id = int(fields[0])
    p_sist = float(fields[1])
    p_diast = float(fields[2])
    q_pa = float(fields[3])
    pulso = float(fields[4])
    respiracao = float(fields[5])
    gravidade = float(fields[6])
    rotulo = int(fields[7]) if len(fields) > 7 else -1
    
    return Exemplo(id, p_sist, p_diast, q_pa, pulso, respiracao, gravidade, rotulo)

def read_data(file_path: str) -> list[Exemplo]:
    """
    Read data from a file and return a list of Exemplo objects.
    """
    
    exemplos = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                exemplo = parse_exemplo(line)
                exemplos.append(exemplo)
    
    return exemplos