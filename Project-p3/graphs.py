import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

def grouped_bar(results, cat_names):
  x = np.arange(len(cat_names))  # the label locations
  width = 0.2  # the width of the bars
  multiplier = 0

  fig, ax = plt.subplots(layout='constrained')

  for attribute, measurement in results.items():
      offset = width * multiplier
      rects = ax.bar(x + offset, measurement, width, label=attribute)
      ax.bar_label(rects, padding=3)
      multiplier += 1

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Counts')
  ax.set_xlabel('')
  ax.set_title('Grouped Bars for ROUGE scores')
  ax.set_xticks(x + width, cat_names)
  ax.legend(loc='upper right', ncols=1)
  ax.set_ylim(0, 500)

  plt.savefig('Rouge1.png', transparent=True, dpi=600)

def survey(results, category_names):

  labels = list(results.keys())
  data = np.array(list(results.values()))
  print(data)
  data_cum = data.cumsum(axis=1)
  category_colors = plt.colormaps['pink'](
      np.linspace(0.15, 0.85, data.shape[1]))

  fig, ax = plt.subplots(figsize=(9.2, 5))
  ax.invert_yaxis()
  ax.xaxis.set_visible(False)
  ax.set_xlim(0, np.sum(data, axis=1).max())

  for i, (colname, color) in enumerate(zip(category_names, category_colors)):
      widths = data[:, i]
      starts = data_cum[:, i] - widths
      rects = ax.barh(labels, widths, left=starts, height=0.5,
                      label=colname, color=color)

      r, g, b, _ = color
      text_color = 'black'
      ax.bar_label(rects, label_type='center', color=text_color)
  ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
            loc='lower left', fontsize='small')

  plt.savefig("Rouge.png", transparent=True, dpi=600)
  # return fig, ax

def main():
  df = pd.read_csv("Orca_predictions.tsv", sep='\t', dtype={"ROUGE": object})
  subcategories = ['0-.25', '.25-.5', '.5-1']
  df['ROUGE'] = df['ROUGE'].apply(ast.literal_eval)
  # Initialize dictionaries to store values for each key and subcategory
  results = {
      'rouge1':[0, 0, 0],
      'rouge2':[0, 0, 0],
      'rougeL':[0, 0, 0],
      'rougeLsum':[0, 0, 0]
  }

  # Iterate over the DataFrame column and aggregate values for each key and subcategory
  for idx, row in df.iterrows():
      try:
        for key, value in row['ROUGE'].items():
            if value <= 0.25:
                results[key][0] += 1
            elif 0.25 < value <= 0.5:
                results[key][1] += 1
            elif 0.5 < value <= 1:
                results[key][2] += 1
      except:
        # continue
        print(idx, " has no scores")
  print(results)
  # Calculate averages for each subcategory and update the 'Average' subcategory
  # for key, subcat_dict in results.items():
  #     for subcat, values in subcat_dict.items():
  #         if values:
  #             avg = np.mean(values)
  #             results[key][subcat] = avg

  survey(results, subcategories)
  grouped_bar(results, subcategories)

if __name__ == "__main__":
    main()
