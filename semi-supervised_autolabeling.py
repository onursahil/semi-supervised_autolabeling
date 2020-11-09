warnings.filterwarnings('ignore')
print("Auto-Labeling Starts...")
# sample_df = sample_df.sample(size)
init_sample_len = len(sample_df)
pre_size = 0
m = 0
round = 0
while pre_size != len(sample_df) != 0:
    round += 1
    start = time.time()
    # sample_df = sample_df.sample(len(sample_df))
    pre_size = len(sample_df)

    train_df = sample_df.copy()

    if init_sample_len != pre_size:
        temp_df = pd.DataFrame()
        max_count = max(sample_df.groupby('topic').count()['token'])
        for topic in topic_list:
            sub_train_df = pd.DataFrame()
            valuable_count = int(max_count / (train_df[train_df['topic'] == topic]['token'].count()))
            if train_df[train_df['topic'] == topic]['token'].count() <= max_count:
                while True:
                    sub_train_df = pd.concat([sub_train_df, train_df[train_df['topic'] == topic].head(max_count)], ignore_index=False)
                    if sub_train_df[sub_train_df['topic'] == topic]['token'].count() >= max_count:
                        break
            temp_df = pd.concat([temp_df, sub_train_df[sub_train_df['topic'] == topic].head(max_count)], ignore_index=False)

        temp_df = temp_df.sample(len(temp_df))
        train_df = temp_df.copy()


    n_estimators = int(len(train_df) / 60)
    print("Number of Estimators: ", n_estimators)
    m += 1

    #     n_samples = int(len(sample_df) / n_estimators) #150 # int(len(sample_df) / n_estimators)
    n_samples = int(len(train_df) / n_estimators)  # 150 # int(len(sample_df) / n_estimators)
    print("Number of samples: ", n_samples)

    nb_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                       ('tfidf', TfidfTransformer()),
                       #                        ('chi', SelectKBest(chi2, k=int(len(sample_df) / n_estimators))),
                       ('chi', SelectPercentile(chi2, percentile=80)),
                       ('clf', MultinomialNB()),
                       ])
    result_df = test_df.copy()


    train_df['topic'] = train_df['topic'].astype('int')
    for count in range(n_estimators):
        sub_train_df = train_df.iloc[n_samples * count:n_samples * (count + 1)]
        nb_clf.fit(sub_train_df['token'], sub_train_df['topic'])
        result_df['nb%d' % count] = nb_clf.predict(result_df['token'])

    print("Result DataFrame: ", "\n", result_df)

    for topic in topic_list:
        summary1_df = test_df[(result_df[['nb%d' % i for i in range(n_estimators)]] == topic).all(axis=1)]
        print("Pre-Summary DataFrame: ", "\n", summary1_df)
        summary1_df.drop(list(set(summary1_df.index) & set(sample_df.index)), inplace=True)
        print("Post-Summary DataFrame: ", "\n", summary1_df)
        summary1_df['topic'] = topic
        sample_df = pd.concat([sample_df, summary1_df], ignore_index=False)

    end = time.time()

    print(
        'Auto-Labeling round %d' %round,
        'Time Consumed: %.2f' %(int(end-start)/60),
        pre_size, '->', len(sample_df),
        '( +',
        len(sample_df) - pre_size,
        ')  Coverage:',
        '%3.2f' % (len(sample_df) / len(test_df) * 100),
        '(%3.2f)' % (np.sum(test_df.loc[sample_df.index, 'topic'] == sample_df['topic']) / len(sample_df['topic']) * 100),
        'Num. Estimator: ', n_estimators,
    )



    sample_df.to_excel('rating_result.xlsx', encoding='utf-8', engine='xlsxwriter')

    # original_data = pd.read_excel('rating_train_tokenize_preview.xlsx', encoding='utf-8')
    # new_sample = pd.read_excel('rating_result.xlsx', encoding='utf-8')
    # correct_labels = []
    # for i in range(len(new_sample)):
    #     content = new_sample.iloc[i][1]
    #     for j in range(len(new_sample) + 1):
    #         if content == original_data.iloc[j][3]:
    #             correct_labels.append(original_data.iloc[j][2])
    # new_sample['correct_labels'] = correct_labels

    # sample_df.to_excel('compare_rating_result.xlsx', encoding='utf-8', engine='xlsxwriter')