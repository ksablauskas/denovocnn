class baseEncoder:
    EMPTY = 0
    A = 1
    C = 2
    T = 3
    G = 4
    IN_A = 5
    IN_C = 6
    IN_T = 7
    IN_G = 8
    DEL = 9

class baseDecoder:
    EMPTY = ' '
    A = 'A'
    C = 'C'
    T = 'T'
    G = 'G'
    IN_A = 'A'
    IN_C = 'C'
    IN_T = 'T'
    IN_G = 'G'
    DEL = '_'

class VariantInheritance:
    DNM = 0
    PV = 1
    MV = 2
    PV_MV = 3
    IV = 1 #Inherited variant

class VariantClassValue:
    snp = 0
    deletion = 1
    insertion = 2
    unknown = 3