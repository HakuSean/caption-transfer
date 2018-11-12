import argparse                                                                  

parent_parser = argparse.ArgumentParser(add_help=False)                                 
parent_parser.add_argument('-H', '--host', default='192.168.122.1')                     
parent_parser.add_argument('-P', '--port', default='12345')                             

parser = argparse.ArgumentParser(add_help=False) 
subparsers = parser.add_subparsers()                                             

# subcommand a                                                                   
parser_a = subparsers.add_parser('a', parents = [parent_parser])                          
parser_a.add_argument('-D', '--daemon', action='store_true')                     

parser_a.add_argument('-L', '--log', default='/tmp/test.log')                    

# subcommand b                                                                   
parser_b = subparsers.add_parser('b', parents = [parent_parser])                          
parser_b.add_argument('-D', '--daemon', action='store_true')                     

# subcommand c                                                                   
parser_c = subparsers.add_parser('c', parents = [parent_parser])                          
args = parser.parse_args()                                                       
