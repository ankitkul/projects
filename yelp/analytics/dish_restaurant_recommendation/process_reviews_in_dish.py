import csv
import json
import pickle

path2files="../task1/yelp_dataset_challenge_academic_dataset/"
path2buisness=path2files+"yelp_academic_dataset_business.json"
path2reviews=path2files+"yelp_academic_dataset_review.json"

def top_dishes():
	top_dishes = []
	with open('data/original_top-dishes.csv') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			top_dishes.append((row[0], row[1]))

	with open ("output/Indian.json" , 'r') as f:
		reviews = json.load(f)
	
	with open ("output/review_rating_indian.json" , 'r') as f:
		review_rating = json.load(f)	

	dish_ratings = {}	
	for key, value in reviews.iteritems():
		for item in top_dishes:
			if item[0] in value:
				id = item[0]+'-'+item[1]
				if id in dish_ratings:
					dish_ratings[id].append(int(review_rating[key]))
				else:
					dish_ratings[id] = [int(review_rating[key])]			 

	popular_dish =[]				
	for k, v in dish_ratings.iteritems():
		total_rating = 0
		for rating in v:
			total_rating += rating

		avg_rating = total_rating * 1 /len(v)
		dish_name = k.split('-')[0]
		dish_freq = int(k.split('-')[1])

		popular_dish.append((dish_name, dish_freq*avg_rating, avg_rating, dish_freq))
	
	with open('output/popular_dish_w_ratings.csv','wb') as out:
	    csv_out=csv.writer(out)
	    csv_out.writerow(['id','value','rating','freq'])
	    for row in popular_dish:
	        csv_out.writerow(row)
	        

def generate_data():
	with open('data/data_cat2rid.pickle', 'rb') as input_file:
	    cat2rid = pickle.load(input_file)

	with open('data/data_rest2revID.pickle', 'rb') as f:
	    rest2revID = pickle.load(f)

	print "sampling categories"
	sample_rid2cat={}

	cat = "Indian"
	for rid in cat2rid[cat]:
	    if rid in rest2revID:
	        if rid not in sample_rid2cat:
	            sample_rid2cat[rid] = []
	        sample_rid2cat[rid].append(cat)
	#remove from memory
	rest2revID=None
	#    print (len(sample_rid2cat), len(cat2rid), len(valid_cats), len(cat_sample))

	print "reading from reviews file..."
	#ensure categories is a directory
	sample_cat2reviews={}
	sample_cat2ratings={}
	business_id = {}
	num_reviews = 0
	with open (path2reviews, 'r') as f:
	    for line in f.readlines():
	        review_json = json.loads(line)
	        rid = review_json['business_id']
	        if rid in sample_rid2cat:
				for rcat in sample_rid2cat [ rid ]:
				    num_reviews = num_reviews + 1
				    sample_cat2reviews[num_reviews] = review_json['text']
				    sample_cat2ratings[num_reviews] = str(review_json['stars'])


				if rid in business_id:
					business_id[rid] += 1
				else:
					business_id[rid] = 1

	print num_reviews
	print len(business_id)    
	print len(sample_cat2reviews)
	print len(sample_cat2ratings)

	print "saving categories"
	with open('output/Indian.json', 'w') as fp:
		json.dump(sample_cat2reviews, fp)

	print "saving ratings for reviews"
	with open('output/review_rating_indian.json', 'w') as fp:
		json.dump(sample_cat2ratings, fp)

	print "saving business_id"
	with open('output/rid_indian.json', 'w') as fp:
		json.dump(business_id, fp)        

generate_data()
#top_dishes()        
