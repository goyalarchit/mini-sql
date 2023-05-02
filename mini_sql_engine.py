import sys
import csv
from operator import itemgetter
import sqlparse
from sqlparse.tokens import Keyword, DML,Wildcard,Whitespace,Punctuation,Comparison as CompOp,Literal
from sqlparse.sql import IdentifierList, Identifier,Where,Function,Comparison
from itertools import product

MetaData = "./metadata.txt"
Files = "./"
Fmt = ".csv"
aggregate_funs = ["sum", "average", "max", "min","count"]
sort_order = ["asc","desc"]
rel_ops = ["<" , ">", "<=", ">=", "="]
Int_Lit = '<int_lit>'

def error(message):
    print("\u001b[31;1mError : \u001b[0m{}".format(message))
    exit(-1)


db = {}

def init_database():
    with open(MetaData, "r") as f:
        contents = f.read().splitlines()
    # print(contents)
    schema = {} 
    for c in contents :
        if c == "<begin_table>" : 
            table_name = None
            attrs = []
        elif c == "<end_table>" :
            schema[table_name] = attrs
        elif table_name == None :
            table_name = c
        else :
            attrs.append(c)
    db['schema'] = schema
    data = load_tables(schema)
    db['data'] = data
    # print(db)


def load_tables(schema) :
    data={}
    for k,v in schema.items():
        with open( Files+k+Fmt, newline='') as csvFile :
            spamreader = list(csv.reader(csvFile,dialect='excel',delimiter=',',quotechar='"' ))
        db_table = list (map (to_int_tuple,spamreader))
        db_table = sorted (db_table,key = itemgetter(1))
        data[k]=db_table
    return data



def to_int_tuple(t):
    return tuple( map(int,t) )


def verify_col(col):
    schema = db['schema']
    for table in schema :
        if col.casefold() in (col_name.casefold() for col_name in schema[table]) :
            return (table,col)
    error("unknown Attribute : {}".format(col))


def verify_table(table):
    schema = db['schema']
    if table in schema :
        return table
    error("Table '{}' does not exist".format(table))


def ast_comparison(tokens):
    comp = {'operands':[]}
    tokens = list(map(lambda x : (x,False),tokens))
    for i,(token,status) in enumerate(tokens):
        if token.ttype == CompOp :
            comp["op"] = token.value
        elif token.ttype == Literal.Number.Integer :
            comp["operands"].append((Int_Lit,token.value))
        elif type(token) is Identifier:
            comp["operands"].append(verify_col(token.value))
        elif token.ttype is Whitespace:
            pass
        else :
            error(" unknown Operator : {} in {}".format(token.value,token.parent.value))
    # print(comp)
    if("op" in comp and len(comp["operands"])==2 ):
        return comp
    else :
        error("Invalid Relational Expression {}".format(tokens[0].parent.value))
            

def ast_where(where_query):
    where={"expr":[]}
    tokens = where_query[0].tokens
    # print(tokens)
    tokens = list(map(lambda x : (x,False),tokens))
    for i,(token,status) in enumerate(tokens):
        if token.ttype == Keyword :
            tokens[i] = (token,True)
            if token.value.upper() == "AND" :
                where["logical"] = "AND"
            elif token.value.upper() == "OR" :
                where["logical"] = "OR"
            elif token.value.upper() == "WHERE" :
                pass
            else :
                error("Not A valid keyword {}".format(token.value))
        elif type(token) is Comparison :
            where["expr"].append(ast_comparison(token))
        elif type(token) is Identifier:
            error("Invalid syntax near '{}'".format(token.value))
        else:
            pass
    if "logical" in where :
        if len(where["expr"])==2 :
            return where
        else :
            error("Need More Arguements for '{}' in \"{}\"".format(where["logical"],where_query[0].value))
    else :
        if len(where["expr"])==1 :
            return where
        elif len(where["expr"]) == 2:
            error("Need 'Logical Operator' for '{}'".format(where_query[0].value))
        else :
            error("Need Expresson for '{}'".format(where_query[0].value))
    
        
def get_index_token(tokens, ttype, value):
    for i,(token,status) in enumerate(tokens):
        if type(token) is ttype :
            return i
        if token.ttype is ttype and token.value.upper() == value:
            return i
    return None

def ast_project(tokens,start,end):
    cols=[]
    distinct = False

    for i in range(start,end):
        token = tokens[i][0]
        if token.ttype is Keyword and token.value.upper() == "SELECT":
            tokens[i] = (token,True)
        elif type(token) is Identifier:
            cols.append(verify_col(token.value)+('',))
            tokens[i] = (token,True)
        elif type(token) is Function:
            fun = token.value[:-1].split("(")
            if fun[1] == '*':
                if fun[0].lower() != 'count':
                    error("Invalid Aggregate function : '{}' on '*'".format(fun[0]))
                else:
                    cols.append(('*','*',fun[0].lower()))
            else:
                if fun[0].lower() not in aggregate_funs:
                    error("No '{}' aggregate Function".format(fun[0]))
                cols.append(verify_col(fun[1])+(fun[0].lower(),))
            tokens[i] = (token,True)
        elif type(token) is IdentifierList :
            for identifier in token.tokens:
                if type(identifier) is Identifier:
                    cols.append(verify_col(identifier.value)+('',))
                elif type(identifier) is Function:
                    fun = identifier.value[:-1].split("(")
                    if fun[1] == '*':
                        if fun[0].lower() != 'count':
                            error("Invalid Aggregate function : '{}' on '*'".format(fun[0]))
                        else:
                            cols.append(('*','*',fun[0].lower()))
                    else:
                        if fun[0].lower() not in aggregate_funs:
                            error("No '{}' aggregate Function".format(fun[0]))
                        cols.append(verify_col(fun[1])+(fun[0].lower(),))
                elif identifier.ttype is Wildcard:
                    cols.append(('*','*',''))        
            tokens[i] = (token,True)
        elif token.ttype is Wildcard:
            cols.append(('*','*',''))
            tokens[i] = (token,True)
        elif token.ttype is Keyword and token.value.upper()=="DISTINCT":
            distinct=True
            tokens[i] = (token,True)
    if len(cols) == 0:
        error("Need Attributes in Select Statement")
    return cols,distinct

def ast_from(tokens,start,end):
    tables = []
    for i in range(start,end):
        token = tokens[i][0]
        if token.ttype is Keyword and token.value.upper() == "FROM":
            tokens[i] = (token,True)
        elif type(token) is Identifier:
            tables.append(verify_table(token.value))
            tokens[i] = (token,True)
        elif type(token) is IdentifierList :
            for identifier in token.tokens:
                if type(identifier) is Identifier :
                    tables.append(verify_table(identifier.value))
            tokens[i] = (token,True)
    
    if len(tables) == 0:
        error("Need Tables in From Statement")

    return tables


def ast_group(tokens,start,end):
    col=None
    for i in range(start,end):
        token= tokens[i][0]
        if token.ttype is Keyword and token.value.upper() == "GROUP BY":
            tokens[i] = (token,True)
        elif type(token) is Identifier :
            col = verify_col(token.value)+('',)
            tokens[i] = (token,True)
            return col
        elif type(token) is Function:
            fun = token.value[:-1].split("(")
            if fun[0].lower() not in aggregate_funs:
                error("No '{}' aggregate Function".format(fun[0]))
            col = (verify_col(fun[1])+(fun[0].lower(),))
            tokens[i] = (token,True)
            return col
    return None


def ast_select(tokens,start,end):
    select={}
    if tokens[start][0].ttype is DML and tokens[start][0].value.upper() == "SELECT":
        tokens[start]=(tokens[start][0],True)
    else :
        error("Invalid Query: Need Select statement")
    from_idx = get_index_token(tokens,Keyword,"FROM")
    if from_idx is None :
        error("No 'FROM' in query '{}'".format(tokens[start][0].parent.value))
    # tokens[from_idx] = (tokens[from_idx][0],True)
    # print(tokens)
    where_idx = get_index_token(tokens,Where,"WHERE")
    if where_idx : tokens[where_idx]=(tokens[where_idx][0],True)
    group_idx = get_index_token(tokens,Keyword,"GROUP BY")
    select["cols"],select["distinct"] = ast_project(tokens,start,from_idx)
    if where_idx == group_idx == None :
        select["from"] = ast_from(tokens,from_idx,end)
        select["where"] = None
        select["group"] = None
    
    elif where_idx is None and group_idx is not None :
        select["from"] = ast_from(tokens,from_idx,group_idx)
        select["where"] = None
        select["group"] = ast_group(tokens,group_idx,end)
    
    elif where_idx is not None and group_idx is None :
        select["from"] = ast_from(tokens,from_idx,where_idx)
        select["where"] = ast_where(tokens[where_idx])
        select["group"] = None
    
    else:
        select["from"] = ast_from(tokens,from_idx,where_idx)
        select["where"] = ast_where(tokens[where_idx])
        select["group"] = ast_group(tokens,group_idx,end)
    return select
    

def ast_order(tokens,start,end):
    order = {}
    for i in range(start,end):
        token = tokens[i][0]
        if token.ttype is Keyword and token.value.upper() == "ORDER BY":
            tokens[i] = (token,True)
        elif type(token) is Identifier :
            col_order = token.value.split(" ") 
            if len(col_order)>2:
                error("Invalid Syntax near '{}'".format(token.value))
            if len(col_order)==2:
                if col_order[1].lower() not in sort_order:
                    error("'{}' is not a valid sorting order".format(col_order[1]))
                order["col"] = verify_col(col_order[0])+('',)
                order["sort"] = col_order[1].lower()
            else:
                order["col"] = verify_col(col_order[0])+('',)
                order["sort"] = sort_order[0]
            tokens[i] = (token,True)
            return order
    return None



def get_ast(q_str):
    ast = {}
    q_str = q_str.lstrip().rstrip()
    if q_str[-1] != ';':
        error("Invalid Syntax, query not terminated with ';' ")
    else:
        q_str=q_str[:-1]
    q_parsed = sqlparse.parse(q_str)[0].tokens
    q_parsed = list(map(lambda x : (x,False),q_parsed))
    # print(q_parsed)

    order_idx = get_index_token(q_parsed,Keyword,"ORDER BY")
    if order_idx is not None :
        # q_parsed[order_idx] =  (q_parsed[order_idx][0],True)
        ast["select"] = ast_select(q_parsed,0,order_idx)
        ast["order"] = ast_order(q_parsed,order_idx,len(q_parsed))
    else :
        ast["select"] = ast_select(q_parsed,0,len(q_parsed))
        ast["order"] = None
    
    for i,(token,status) in enumerate(q_parsed):
        if token.ttype is Whitespace :
            q_parsed[i] = (token,True);
    

    # print(q_parsed)
    # print(ast)
    if (any(s == False for (t,s) in q_parsed )):
        error("Invalid Query")
    return ast

###Execution###



def execute_from(table_list,instance):
    # print(table_list)
    final_table={}
    all_tables = list(instance["schema"].keys())
    # print(all_tables)
    for table in all_tables:
        if table not in table_list:
            instance["schema"].pop(table)
            instance["data"].pop(table)
    f_data = []
    f_cols = [] 
    for t in instance["schema"]:
        f_data.append(instance["data"][t])
        f_cols = f_cols + instance["schema"][t]
    f_data = list(product(*f_data))
    f_data = list(map(lambda x : tuple(sum(x,())), f_data))
    final_table["cols"]=f_cols
    final_table["data"]=f_data

    return final_table

def col_pos(col_name,header):
    if col_name.casefold() in list(map(lambda x : x.casefold(),header)):
        return (list(map(lambda x : x.casefold(),header)).index(col_name.casefold()))
    else:
        error("Unknown Field : '{}'".format(col_name))

def apply_op(qry,tuple):
    if (eval(qry)):
        return True
    else:
        return False

def map_op(op):
    if op not in rel_ops:
        error("Unknown Operator '{}'".format(op))
    if op == "=":
        return "=="
    else:
        return op

def intersection(data_1,data_2):
    final_data = []
    for t in data_1:
        if t in data_2:
            final_data.append(t)
    return final_data

def diff(data_1,data_2):
    final_data = []
    for t in data_1:
        if t not in data_2:
            final_data.append(t)
    return final_data

def execute_where(where_qry,table):
    if where_qry is None:
        return table
    data=[]
    for expr in where_qry["expr"]:
        opds=[]
        for opd in expr["operands"]:
            if opd[0] is Int_Lit:
                opds.append(opd[1])
            else:
                opds.append("tuple["+str(col_pos(opd[1],table["cols"]))+"]")
        qry = opds[0]+map_op(expr["op"])+opds[1]
        data.append(list(filter(lambda x:apply_op(qry,x),table["data"])))
    if len(data) == 2:
        if where_qry["logical"] == "AND":
            data = intersection(data[0],data[1])
        elif where_qry["logical"] == "OR":
            data = data[0]+diff(data[1],intersection(data[0],data[1])) 
    else:
        data = data[0]
    table["data"] = data
    return table



def execute_group(para,table):
    if para is None:
        table["group"] = None
        table["data"] = [table["data"]]
        return table
    grouped_table = {}
    col_num = col_pos(para[1],table["cols"])
    data = sorted(table["data"],key=itemgetter(col_num))
    # print(data)
    distinct_val = []
    grouped_data = []
    group = []
    for t in data:
        if t[col_num] in distinct_val:
            group.append(t)
        else:
            distinct_val.append(t[col_num])
            if group :
                grouped_data.append(group)
            group = [t]
    if group :
        grouped_data.append(group)
    grouped_table["cols"] = table["cols"]
    grouped_table["data"] = grouped_data
    grouped_table["group"] = col_num
    return grouped_table 




def apply_aggr(fun,col,table):
    op = []
    if col != '*':
        col_num = col_pos(col,table["cols"])
    if fun not in aggregate_funs:
        error("Unknown Aggregate Function '{}'".format(fun))
    for group in table["data"]:
        req_data = group
        if col != '*':
            req_data = [t[col_num] for t in group]
        if fun == 'sum':
            op.append(sum(req_data))
        elif fun == 'average':
            op.append(sum(req_data)/len(req_data))
        elif fun == 'max':
            op.append(max(req_data))
        elif fun == 'min':
            op.append(min(req_data))
        elif fun == 'count':
            op.append(len(req_data))
    # print(len(op),len(table["data"]))
    return op
        

        
            


def execute_aggr_func(cols,table):

    aggr_func = {}
    aggr_func_data = []
    aggr_func_group = []
    for col in cols:
        if col[2] != '':
            aggr_func[col[2]+'('+col[1]+')'] = len(aggr_func_data)
            aggr_func_data.append(apply_aggr(col[2],col[1],table))
    table["aggr"]=aggr_func
    table["aggr_data"] = aggr_func_data
    return table

def get_col_header(col):
    return (col[0]+'.'+col[1])

def get_all_col_header(cols):
    all_cols=[]
    for col_name in cols :
        col=verify_col(col_name)
        all_cols.append(get_col_header(col))
    return all_cols


def remove_add_cols(table):
    final_table = {}
    l = len(table["cols"])
    final_data = []
    for t in table["data"]:
        final_data.append(t[l:])
    final_table["header"]=table["header"]
    final_table["data"] = final_data
    return final_table


def execute_order(order,table):
    if order is None:
        return (remove_add_cols(table))
    col_num = col_pos(order["col"][1],table["cols"])
    if order["sort"] == 'desc' : reverseflag = True
    else : reverseflag = False
    ordered_data = sorted(table["data"],key=itemgetter(col_num),reverse=reverseflag)
    table["data"] = ordered_data
    # print(ordered_data)
    return (remove_add_cols(table))

def execute_select(cols,table):
    project_table = {}
    op_data = []
    header = []
    all_cols= get_all_col_header(table["cols"])
    for col in cols:
        if col[1] == '*' and col[2] == '' :
            header=header+all_cols
        elif col[2] == '':
            header.append(get_col_header(col))
        else:
            header.append(col[2]+'('+col[1]+')')
    if table["aggr"]:
        for i,group in enumerate(table["data"]):
            gp_op = group[0]
            for col in cols:
                if col[1] == '*' and col[2]=='':
                    gp_op = gp_op+group[0]
                elif col[2] == '':
                    gp_op = gp_op+(group[0][col_pos(col[1],table["cols"])],)
                else:
                    aggr_idx = table["aggr"][col[2]+'('+col[1]+')']
                    gp_op = gp_op+(table["aggr_data"][aggr_idx][i],)
            op_data.append(gp_op)
    elif table["group"] is not None:
        for i,group in enumerate(table["data"]):
            gp_op = group[0]
            for col in cols:
                if col[1] == '*' and col[2]=='':
                    gp_op = gp_op+group[0]
                else :
                    gp_op = gp_op+(group[0][col_pos(col[1],table["cols"])],)
            op_data.append(gp_op)
    else :
        for i,t in enumerate(table["data"][0]):
            gp_op = t
            for col in cols:
                if col[1] == '*':
                    gp_op = gp_op+t
                else :
                    gp_op = gp_op+(t[col_pos(col[1],table["cols"])],)
            op_data.append(gp_op)
    project_table["cols"] = table["cols"]
    project_table["header"]=header
    project_table["data"] = op_data
    return project_table
    
def print_row(row_data):
    row_data = list(map(lambda x : x.lower() if isinstance(x,str) else x,row_data ))
    # print(new_data)
    for cell in row_data[:-1]:
        print(cell,end=",")
    print(row_data[-1])

def execute_distinct(data_tuple):
    distinct_data = [] 
    for t in data_tuple: 
        if t not in distinct_data:
            distinct_data.append(t)
    return distinct_data

def print_table(distinct,table):
    data = table["data"] 
    if distinct:
        data = execute_distinct(table["data"])
    print_row(table["header"])
    for t in data:
        print_row(t) 


def execute(ast,instance):
    from_table = execute_from(ast["select"]["from"],instance)
    # print(from_table)
    where_table= execute_where(ast["select"]["where"],from_table)
    # print(where_table)
    group_table = execute_group(ast["select"]["group"],where_table)
    # print(group_table)
    aggr_table = execute_aggr_func(ast["select"]["cols"],group_table)
    # print(aggr_table)
    select_table = execute_select(ast["select"]["cols"],aggr_table)
    # print(select_table)
    order_table = execute_order(ast["order"],select_table)
    # print(order_table)
    print_table(ast["select"]["distinct"],order_table)








def main():
    if len(sys.argv)!=2:
        print("Error : Incorrect Format")
        exit(-1)
    else :
        # print(sys.argv)
        init_database()
        q = sys.argv[1]
        ast = get_ast(q)
        execute(ast,db)
        # print(ast)
    

if __name__ == "__main__" :
    main()





