#old code:
# probably delete this





documents = {f: re.split('\n\n(?=\u2028|[A-Z-0-9])',
                [parser.from_file(DATA_PATH + f + '/' + doc) for doc in os.listdir(DATA_PATH + f)]
            [0]['content'].strip()
          )
      for f in files[8:9]
     }