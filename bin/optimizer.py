# optimizer
# binding.pry equivalent:
# import code; code.interact(local=dict(globals(), **locals()))

import math
import datetime
import os, sys
import argparse
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

from timeit import default_timer as timer

####################
#                  #
#  HELPER METHODS  #
#                  #
####################

start = timer()
last_call = start
def elapsed_time():
  global last_call
  next_last_call = timer()
  elapsed = next_last_call - last_call
  last_call = next_last_call

  return elapsed

class style():
  BLACK = lambda x: '\033[30m' + str(x) + '\033[0m'
  RED = lambda x: '\033[31m' + str(x) + '\033[0m'
  GREEN = lambda x: '\033[32m' + str(x) + '\033[0m'
  YELLOW = lambda x: '\033[33m' + str(x) + '\033[0m'
  BLUE = lambda x: '\033[34m' + str(x) + '\033[0m'
  MAGENTA = lambda x: '\033[35m' + str(x) + '\033[0m'
  CYAN = lambda x: '\033[36m' + str(x) + '\033[0m'
  WHITE = lambda x: '\033[37m' + str(x) + '\033[0m'
  UNDERLINE = lambda x: '\033[4m' + str(x) + '\033[0m'

def as_date(date_string):
  if date_string is None: return None

  try:
    return datetime.datetime.strptime(date_string, '%Y-%m-%d')
  except ValueError:
    return None

def as_pct(float_string):
  if float_string is None: return None

  try:
    return float(float_string) / 100.0
  except ValueError:
    return None

def as_int(int_string):
  if int_string is None: return None

  try:
    return int(int_string)
  except ValueError:
    return None

def months_from_date(start_date, months_offset):
  year, month, day = start_date.timetuple()[:3]
  new_month = month + months_offset

  return datetime.date(year + math.floor(new_month / 12), (new_month % 12) or 12, day)

#########
#       #
#  I/O  #
#       #
#########

def progress(msg):
  print(style.BLACK('({:.3f} sec) {}...'.format(elapsed_time(), msg)))

def get_inputs():
  all_args_provided = True

  parser = argparse.ArgumentParser(description='Portfolio Optimization')
  parser.add_argument(
    '--cutoff_date',
    type=str,
    help='Cutoff Date for Analysis (ex: 2013-01-01)'
  )
  parser.add_argument(
    '--weight_bound',
    type=str,
    help='Max percentage for each security (0.0-100.0)'
  )
  parser.add_argument(
    '--return_months',
    type=str,
    help='Forward looking months to calculate returns (ex: 6, 12)'
  )

  args = parser.parse_args()

  args.cutoff_date = as_date(args.cutoff_date)
  args.weight_bound = as_pct(args.weight_bound)
  args.return_months = as_int(args.return_months)

  while not args.cutoff_date:
    all_args_provided = False
    user_input_date = input(style.MAGENTA("Cutoff Date for Analysis (ex: 2013-01-01): "))
    args.cutoff_date = as_date(user_input_date)

  while (not args.weight_bound) or args.weight_bound < 0.0 or args.weight_bound > 1.0:
    all_args_provided = False
    user_input_weight = input(style.MAGENTA("Max percentage for each security (0.0-100.0): "))
    args.weight_bound = as_pct(user_input_weight)

  while (not args.return_months) or args.return_months < 0:
    all_args_provided = False
    user_input_return_months = \
      input(style.MAGENTA("Forward looking months to calculate returns (ex: 6, 12): "))
    args.return_months = as_int(user_input_return_months)

  return args, all_args_provided

def print_as_full_command_line_statement(args):
  command_line_args = ' '.join([
    '--cutoff_date {}'.format(args.cutoff_date.strftime("%Y-%m-%d")),
    '--weight_bound {:.2f}'.format(args.weight_bound * 100.0),
    '--return_months {}'.format(str(args.return_months))
  ])
  command = 'python3 {} {}'.format(
    sys.argv[0],
    command_line_args
  )
  print('PROTIP: skip the prompts with: {}'.format(style.YELLOW(command)))

#######################
#                     #
#  OPTIMIZER METHODS  #
#                     #
#######################

def read_pricing():
  progress('Loading Pricing Data')
  return pd.read_csv(
    "tests/MLP Prices - Adjusted Prices.csv",
    parse_dates=True,
    index_col="date"
  )

def generate_analysis_model(pricing_data, start_date, end_date, upper_bound):
  progress('Processing Pricing')
  df_lookback = pricing_data.loc[start_date:end_date].dropna(axis=1, how='any')
  df = pricing_data.loc[:end_date].filter(axis=1, items=df_lookback.columns)

  # Calculate expected returns and sample covariance
  mu = expected_returns.mean_historical_return(df, 12)
  S = risk_models.sample_cov(df_lookback, 12)

  range_begin = df_lookback.index[0]
  range_end = df_lookback.index[-1]

  return range_begin, range_end, EfficientFrontier(mu, S, weight_bounds=(0, upper_bound))

def forward_looking_return(pricing_data, start_date, end_date, weights):
  # get first and last row within the date range
  df_pricing = pricing_data \
    .loc[start_date:end_date] \
    .dropna(axis=1, how='any') \
    .iloc[[0, -1]]

  df_weights = pd.DataFrame([weights])
  pricing_changes = df_pricing.pct_change()
  filtered_pricing = pricing_changes.tail(1).filter(axis=1, items=weights.keys())

  range_begin, range_end = pricing_changes.index.values
  wavg_return = (df_weights.values * filtered_pricing.values).sum()

  return range_begin, range_end, wavg_return

def generate_report(pricing_data, start_date, end_date, weight_bound, returns_start_date, returns_end_date):
  analysis_range_begin, analysis_range_end, ef = \
    generate_analysis_model(pricing_data, start_date, end_date, weight_bound)
  # raw_weights = ef.max_sharpe()
  # Optimise for maximal Sharpe ratio
  progress('Choosing securities')
  _raw_weights = ef.min_volatility()
  cleaned_weights = ef.clean_weights()
  progress('Calculating sharpe ratios')
  _mu, sigma, sharpe = ef.portfolio_performance()

  non_zero_weights = dict(
    filter(lambda w: w[1] > 0.0, cleaned_weights.items())
  )

  progress('Calculating Returns')
  range_begin, range_end, wavg_return = forward_looking_return(
    pricing_data,
    returns_start_date,
    returns_end_date,
    non_zero_weights
  )

  # print("Expected annual return: {:.1f}%".format(100 * mu))
  print(style.GREEN("Portfolio Return ({}...{}): {:.3f}%".format(str(range_begin)[:10], str(range_end)[:10], 100 * wavg_return)))
  print(style.GREEN("Annual volatility: {:.1f}%".format(100 * sigma)))
  print(style.GREEN("Sharpe Ratio: {:.2f}".format(sharpe)))

  for key, value in sorted(non_zero_weights.items(), key=lambda kv: -kv[1]):
    print("{}\t{:.3f}%".format(key, 100 * value))

  print(style.BLACK('{}...{}'.format(analysis_range_begin.strftime('%Y-%m-%d'), analysis_range_end.strftime('%Y-%m-%d'))))

##########
#        #
#  MAIN  #
#        #
##########

def main():
  args, provided_by_command_line = get_inputs()

  end_date = args.cutoff_date - datetime.timedelta(days=10) # attempt to ignore the right fencepost, assumes monthly data
  start_date = months_from_date(end_date, -36)

  returns_start_date = args.cutoff_date - datetime.timedelta(days=10) # attempt to include cutoff_date if it is a fencepost
  returns_end_date = months_from_date(end_date, args.return_months)

  print(style.BLUE("Timeframe ending on or before {}, Cap: {:.2f}%".format(args.cutoff_date.strftime('%Y-%m-%d'), args.weight_bound * 100.0)))

  pricing_data = read_pricing()

  generate_report(
    pricing_data,
    start_date,
    end_date,
    args.weight_bound,
    returns_start_date,
    returns_end_date
  )

  if not provided_by_command_line:
    print_as_full_command_line_statement(args)

main()
print(style.BLACK('(done in {:.3f} seconds)'.format(datetime.timedelta(seconds=elapsed_time()).total_seconds())))
