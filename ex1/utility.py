import pandas as pd
import json
from collections import deque
import heapq

from IPython.display import display, HTML
from functools import total_ordering

debug = True
def pretty_print(df):
    return display( HTML( df.to_html().replace("\\n","<br>")))

def format_search_log(log,limit=100):
  columns=['node to expand', 'path', 'left in frontier', 'valid actions','is_goal']
  return pretty_print(pd.DataFrame(log, columns=columns).head(limit))

def format_search_with_costs_log(log, limit=100):
  columns=['node to expand', 'path', 'g','f','left in frontier', 'valid actions','is_goal']
  return pretty_print(pd.DataFrame(log, columns=columns).head(limit))

# dict on python 3.7+ preserves insertion order.
# This is a quick way to create a set which preserves it as well.
# required for presentation purposes only.
def ordered_set(coll):
  return dict.fromkeys(coll).keys()

def swap_tuple(a_tuple, i, j):
  l = list(a_tuple)
  l[i],l[j] = l[j],l[i]
  return tuple(l)

def load_routes(routes, symmetric=True):
  def insert(frm,to,cost):
    if frm in G:
      G[frm][to]=cost
    else:
      G[frm] = {to:cost}
  G = {}
  routes = routes.splitlines()
  for route in routes:
    r = route.split(',')
    insert(r[0],r[1],int(r[2]))
    if symmetric:
      insert(r[1],r[0],int(r[2]))
  return G