class Exemplo:
    def __init__ (self, id: int, p_sist: float, p_diast: float, q_pa: float, pulso, respiracao, gravidade, rotulo: str = ""):
        self.id = id
        self.p_sist = p_sist
        self.p_diast = p_diast
        self.q_pa = q_pa
        self.pulso = pulso
        self.respiracao = respiracao
        self.gravidade = gravidade
        self.rotulo = rotulo
        
def parse_exemplo(line: str) -> Exemplo:
    """
    Parse a line of data into an Exemplo object.
    """
    fields = line.strip().split(",")
    id = int(fields[0])
    p_sist = float(fields[1])
    p_diast = float(fields[2])
    q_pa = float(fields[3])
    pulso = int(fields[4])
    respiracao = int(fields[5])
    gravidade = int(fields[6])
    rotulo = fields[7] if len(fields) > 7 else ""
    
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