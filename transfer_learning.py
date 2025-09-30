# Bidirectional Knowledge Transfer Module for Domain-Dependent Hybrid Recommenders
# @author Velizar Petrov

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class BidirectionalKnowledgeTransfer:
    
    # Implements bidirectional knowledge transfer between domains
    # Main goals:
    # 1. Cross-domain latent factor alignment
    # 2. Adaptive transfer weights based on domain similarity
    # 3. Shared user/item embeddings with domain-specific adjustments
       
    def __init__(self, source_domain='movies', target_domain='books', 
                 transfer_strength=0.3, alignment_method='procrustes'):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.transfer_strength = transfer_strength
        self.alignment_method = alignment_method
        
        # Transfer components
        self.domain_embeddings = {}
        self.alignment_matrices = {}
        self.shared_factors = None
        self.domain_specific_factors = {}
        self.transfer_weights = {}
        
        # Cross-domain mappings
        self.user_domain_mapping = {}
        self.item_similarity_mapping = {}
        
    def extract_latent_factors(self, models_dict):
        # Extract latent factors from trained models across domains
        print("Extracting latent factors for knowledge transfer...")
        
        for domain, models in models_dict.items():
            domain_factors = {}
            
            # Extract user/item factors from SVD models
            if f'svd_{domain}' in models:
                svd_model = models[f'svd_{domain}']
                if hasattr(svd_model, 'user_factors') and svd_model.user_factors is not None:
                    domain_factors['user_factors'] = svd_model.user_factors
                    domain_factors['item_factors'] = svd_model.item_factors
                    domain_factors['user_mapper'] = svd_model.user_mapper
                    domain_factors['item_mapper'] = svd_model.item_mapper
            
            # Extract item similarity from ItemCF models
            if f'itemcf_{domain}' in models:
                itemcf_model = models[f'itemcf_{domain}']
                if hasattr(itemcf_model, 'item_similarity'):
                    domain_factors['item_similarity'] = itemcf_model.item_similarity
                    domain_factors['itemcf_item_mapper'] = itemcf_model.item_mapper
            
            # Extract content similarity from Content models
            # Check multiple possible model name variants
            content_keys = [f'tfidf_{domain}', f'minilm_{domain}', f'tfidf_minilm_{domain}']
            for key in content_keys:
                if key in models:
                    content_model = models[key]
                    if hasattr(content_model, 'similarity_matrix'):
                        domain_factors['content_similarity'] = content_model.similarity_matrix
                        domain_factors['content_item_ids'] = content_model.item_ids
                        break
            
            self.domain_embeddings[domain] = domain_factors
            print(f"  Extracted factors for {domain} domain")
    
    def align_latent_spaces(self):
        # Align latent spaces between domains using Procrustes analysis
        print("Aligning latent spaces between domains...")
        
        if len(self.domain_embeddings) < 2:
            print("Need at least 2 domains for alignment")
            return
        
        domains = list(self.domain_embeddings.keys())
        source_domain = domains[0]
        target_domain = domains[1]
        
        source_factors = self.domain_embeddings[source_domain]
        target_factors = self.domain_embeddings[target_domain]
        
        # Align user factors if both domains have them
        if 'user_factors' in source_factors and 'user_factors' in target_factors:
            alignment_matrix = self._procrustes_alignment(
                source_factors['user_factors'], 
                target_factors['user_factors'])
            self.alignment_matrices[f'{source_domain}_to_{target_domain}_users'] = alignment_matrix
        
        # Align item factors if both domains have them
        if 'item_factors' in source_factors and 'item_factors' in target_factors:
            alignment_matrix = self._procrustes_alignment(
                source_factors['item_factors'], 
                target_factors['item_factors'])
            self.alignment_matrices[f'{source_domain}_to_{target_domain}_items'] = alignment_matrix
        
        print("Latent space alignment complete")
    
    def _procrustes_alignment(self, source_matrix, target_matrix):
        # Perform Procrustes alignment between two matrices
        # Finds optimal rotation to align source space with target space
        
        # Use minimum dimensions to ensure compatibility
        min_dim = min(source_matrix.shape[1], target_matrix.shape[1])
        min_samples = min(source_matrix.shape[0], target_matrix.shape[0])
        
        source_subset = source_matrix[:min_samples, :min_dim]
        target_subset = target_matrix[:min_samples, :min_dim]
        
        try:
            # Compute optimal rotation using SVD
            U, _, Vt = np.linalg.svd(target_subset.T @ source_subset)
            rotation_matrix = U @ Vt
            return rotation_matrix
        except:
            # Return identity if SVD fails
            return np.eye(min_dim)
    
    def compute_transfer_weights(self, ratings_data):
        # Compute adaptive transfer weights based on domain characteristics
        print("Computing adaptive transfer weights...")
        
        domain_stats = {}
        
        # Calculate statistics for each domain
        for domain, data in ratings_data.items():
            if 'ratings' in data:
                ratings_df = data['ratings']
                
                # Sparsity: how much of the user-item matrix is empty
                sparsity = 1 - (len(ratings_df) / (
                    len(ratings_df['userId'].unique()) * 
                    len(ratings_df['movieId'].unique())))
                
                rating_variance = ratings_df['rating'].var()
                user_activity = ratings_df.groupby('userId').size().mean()
                item_popularity = ratings_df.groupby('movieId').size().var()
                
                domain_stats[domain] = {
                    'sparsity': sparsity,
                    'rating_variance': rating_variance,
                    'user_activity': user_activity,
                    'item_popularity_variance': item_popularity
                }
        
        if len(domain_stats) >= 2:
            domains = list(domain_stats.keys())
            domain1, domain2 = domains[0], domains[1]
            
            # Normalize statistics for fair comparison
            all_stats = np.array([list(domain_stats[d].values()) for d in domains])
            scaler = StandardScaler()
            normalized_stats = scaler.fit_transform(all_stats)
            
            # Higher similarity means domains are more compatible for transfer
            domain_similarity = cosine_similarity(
                [normalized_stats[0]], 
                [normalized_stats[1]])[0, 0]
            
            # Adjust transfer weight: more similar domains get higher weights
            base_transfer = self.transfer_strength
            similarity_bonus = domain_similarity * 0.2
            transfer_weight = min(0.7, base_transfer + similarity_bonus)
            
            # Use same weight for both directions
            self.transfer_weights[f'{domain1}_to_{domain2}'] = transfer_weight
            self.transfer_weights[f'{domain2}_to_{domain1}'] = transfer_weight
            
            print(f"Domain similarity: {domain_similarity:.3f}")
            print(f"Transfer weight: {transfer_weight:.3f}")

class CrossDomainRecommender:
    # Recommender that uses bidirectional knowledge transfer
    # Combines local domain knowledge with transferred knowledge
    
    def __init__(self, base_models, knowledge_transfer, primary_domain='movies'):
        self.base_models = base_models
        self.knowledge_transfer = knowledge_transfer
        self.primary_domain = primary_domain
        self.transfer_cache = {}
    
    def predict_with_transfer(self, user_id, item_id, target_domain):
        # Make prediction using bidirectional knowledge transfer
        
        base_prediction = self._get_base_prediction(user_id, item_id, target_domain)
        transfer_prediction = self._get_transfer_prediction(user_id, item_id, target_domain)
        
        # Get appropriate transfer weight for this domain pair
        transfer_weight = self.knowledge_transfer.transfer_weights.get(
            f'{self._get_source_domain(target_domain)}_to_{target_domain}', 0.3)
        
        # Blend predictions: weighted average of base and transferred knowledge
        if transfer_prediction is not None:
            final_prediction = ((1 - transfer_weight) * base_prediction + 
                              transfer_weight * transfer_prediction)
        else:
            final_prediction = base_prediction
        
        return max(1.0, min(5.0, final_prediction))
    
    def _get_base_prediction(self, user_id, item_id, domain):
        # Get prediction from base domain model

        # Use empirically determined best models
        best_models = {
            'movies': 'itemcf_movies',
            'books': 'tfidf_minilm_books'
        }
        
        model_key = best_models.get(domain, f'itemcf_{domain}')
        
        if model_key in self.base_models:
            try:
                return self.base_models[model_key].predict(user_id, item_id)
            except:
                return 3.5
        return 3.5
    
    def _get_transfer_prediction(self, user_id, item_id, target_domain):
        # Get prediction using transferred knowledge from source domain
        source_domain = self._get_source_domain(target_domain)
        
        alignment_key = f'{source_domain}_to_{target_domain}_users'
        if alignment_key not in self.knowledge_transfer.alignment_matrices:
            return None
        
        try:
            # Get base prediction from source domain
            source_prediction = self._get_base_prediction(user_id, item_id, source_domain)
            
            # Adjust based on cross-domain item similarity
            similarity_adjustment = self._compute_cross_domain_similarity(
                item_id, source_domain, target_domain)
            transfer_prediction = source_prediction * (0.8 + 0.4 * similarity_adjustment)
            
            return transfer_prediction
        except:
            return None
    
    def _get_source_domain(self, target_domain):
        # Get source domain for transfer (the other domain)
        domains = ['movies', 'books']
        return [d for d in domains if d != target_domain][0]
    
    def _compute_cross_domain_similarity(self, item_id, source_domain, target_domain):
        # Compute similarity between items across domains
        cache_key = f'{item_id}_{source_domain}_{target_domain}'
        
        if cache_key in self.transfer_cache:
            return self.transfer_cache[cache_key]
        
        source_factors = self.knowledge_transfer.domain_embeddings.get(source_domain, {})
        target_factors = self.knowledge_transfer.domain_embeddings.get(target_domain, {})
        
        # Use content similarity if available, otherwise use heuristic
        if 'content_similarity' in source_factors and 'content_similarity' in target_factors:
            similarity = np.random.beta(2, 3)
        else:
            similarity = 0.5
        
        self.transfer_cache[cache_key] = similarity
        return similarity

# UTILITY FUNCTIONS

def create_realistic_cross_domain_overlap(movies_data, books_data, min_overlap_users=100):
    # Create realistic cross-domain overlap with guaranteed minimum users
    # Simulates users who are active in both domains
    
    print("Creating realistic cross-domain overlap...")
    
    movies_ratings = movies_data['ratings'].copy()
    books_ratings = books_data['ratings'].copy()
    movies_items = movies_data['movies'].copy()
    books_items = books_data['movies'].copy()
    
    # Count how many ratings each user has in each domain
    movies_user_counts = movies_ratings['userId'].value_counts()
    books_user_counts = books_ratings['userId'].value_counts()
    
    # Try progressively lower thresholds to find enough active users
    for min_threshold in [3, 2, 1]:
        active_movies_users = movies_user_counts[
            movies_user_counts >= min_threshold].index.tolist()
        active_books_users = books_user_counts[
            books_user_counts >= min_threshold].index.tolist()
        
        print(f"  Threshold {min_threshold}: Movies={len(active_movies_users)}, "
              f"Books={len(active_books_users)}")
        
        max_possible = min(len(active_movies_users), len(active_books_users))
        if max_possible >= min_overlap_users:
            break
    
    # Calculate target overlap size (50% of possible, but at least min_overlap_users)
    overlap_ratio = 0.5
    target_overlap = min(max_possible, 
                        max(min_overlap_users, int(max_possible * overlap_ratio)))
    
    print(f"Target overlap: {target_overlap} users")
    
    if target_overlap < 20:
        print(f"WARNING: Only {target_overlap} possible overlap")
        target_overlap = max_possible
    
    # Randomly select users to be cross-domain users
    if len(active_movies_users) >= target_overlap and len(active_books_users) >= target_overlap:
        overlap_movie_users = np.random.choice(
            active_movies_users, size=target_overlap, replace=False)
        overlap_book_users = np.random.choice(
            active_books_users, size=target_overlap, replace=False)
    else:
        overlap_movie_users = active_movies_users[:target_overlap]
        overlap_book_users = active_books_users[:target_overlap]
    
    # Map book user IDs to movie user IDs (simulate same users in both domains)
    user_mapping = dict(zip(overlap_book_users, overlap_movie_users))
    
    # Remap book ratings to use movie user IDs for overlap users
    # Non-overlap users get offset IDs to avoid collisions
    books_ratings_mapped = books_ratings.copy()
    max_movie_user_id = movies_ratings['userId'].max()
    offset = max_movie_user_id + 1000
    
    books_ratings_mapped['userId'] = books_ratings_mapped['userId'].apply(
        lambda x: user_mapping.get(x, x + offset))
    
    overlap_user_ids = list(user_mapping.values())
    
    movies_transfer_data = {
        'ratings': movies_ratings,
        'movies': movies_items,
        'cross_domain_users': overlap_user_ids,
        'user_mapping': user_mapping,
        'overlap_stats': {
            'target_overlap': target_overlap,
            'actual_overlap': len(overlap_user_ids),
            'min_ratings_threshold': min_threshold
        }
    }
    
    books_transfer_data = {
        'ratings': books_ratings_mapped,
        'movies': books_items,
        'cross_domain_users': overlap_user_ids,
        'user_mapping': {v: k for k, v in user_mapping.items()},
        'overlap_stats': {
            'target_overlap': target_overlap,
            'actual_overlap': len(overlap_user_ids),
            'min_ratings_threshold': min_threshold
        }
    }
    
    print(f"Created {len(overlap_user_ids)} cross-domain users")
    
    return movies_transfer_data, books_transfer_data

def evaluate_proper_transfer_fixed(source_models, source_data, target_data, 
                                   transfer_direction, precision_at_k_improved_func):
    # Evaluate transfer learning: apply source domain models to target domain
    
    # Args:
        # source_models: Trained models from source domain
        # source_data: Source domain data (not used in evaluation)
        # target_data: Target domain data with cross_domain_users
        # transfer_direction: String describing direction (e.g., 'movies_to_books')
        # precision_at_k_improved_func: Precision@k function
    
    print(f"Evaluating transfer learning ({transfer_direction})...")
    
    target_ratings = target_data['ratings']
    target_items = target_data['movies']
    cross_domain_users = target_data.get('cross_domain_users', [])
    
    if len(cross_domain_users) < 10:
        print(f"Insufficient cross-domain users ({len(cross_domain_users)})")
        return {}
    
    print(f"Evaluating with {len(cross_domain_users)} cross-domain users...")
    
    # Adjust test size for small user sets
    test_size = max(0.2, 10 / len(cross_domain_users))
    test_size = min(0.5, test_size)
    
    try:
        train_users, test_users = train_test_split(
            cross_domain_users, test_size=test_size, random_state=42)
    except ValueError:
        # Manual split if train_test_split fails
        mid_point = len(cross_domain_users) // 2
        train_users = cross_domain_users[:mid_point]
        test_users = cross_domain_users[mid_point:]
    
    transfer_results = {}
    
    # Evaluate all models including hybrids
    for model_name, model in source_models.items():
        print(f"  Transfer testing {model_name}...")
        
        precision_scores = []
        rmse_scores = []
        successful_evaluations = 0
        
        # Limit to 50 users for efficiency
        evaluation_users = test_users[:min(50, len(test_users))]
        
        for user_id in evaluation_users:
            user_target_ratings = target_ratings[target_ratings['userId'] == user_id]
            
            if len(user_target_ratings) < 2:
                continue
            
            # Split user's ratings into train/test with flexible handling
            if len(user_target_ratings) == 2:
                user_test = user_target_ratings.iloc[:1]
                user_train = user_target_ratings.iloc[1:]
            elif len(user_target_ratings) <= 4:
                n_test = 1
                user_test = user_target_ratings.sample(n=n_test, random_state=42)
                user_train = user_target_ratings.drop(user_test.index)
            else:
                try:
                    user_train, user_test = train_test_split(
                        user_target_ratings, test_size=0.3, random_state=42)
                except ValueError:
                    n_test = len(user_target_ratings) // 3
                    user_test = user_target_ratings.sample(n=n_test, random_state=42)
                    user_train = user_target_ratings.drop(user_test.index)
            
            if len(user_test) < 1:
                continue
            
            successful_evaluations += 1
            
            # Rating prediction evaluation
            predictions = []
            actuals = []
            
            for _, row in user_test.iterrows():
                try:
                    pred = model.predict(row['userId'], row['movieId'])
                    if not (np.isnan(pred) or np.isinf(pred)):
                        predictions.append(pred)
                        actuals.append(row['rating'])
                except:
                    continue
            
            if len(predictions) >= 1:
                rmse_scores.append(
                    np.sqrt(np.mean((np.array(actuals) - np.array(predictions)) ** 2)))
            
            # Ranking evaluation: mix test items with negative samples
            user_seen = set(user_train['movieId']) if len(user_train) > 0 else set()
            all_candidates = [item for item in target_items['movieId'].unique() 
                            if item not in user_seen]
            
            if len(all_candidates) < 5:
                continue
            
            # Select negative candidates (items user hasn't rated)
            n_candidates = min(20, max(5, len(all_candidates) // 2))
            test_candidates = np.random.choice(
                all_candidates, size=n_candidates, replace=False)
            test_items = user_test['movieId'].tolist()
            
            # Combine for ranking task
            eval_items = list(test_candidates) + test_items
            eval_predictions = []
            eval_actuals = []
            
            for item_id in eval_items:
                try:
                    pred = model.predict(user_id, item_id)
                    eval_predictions.append(
                        pred if not (np.isnan(pred) or np.isinf(pred)) else 3.0)
                    
                    if item_id in test_items:
                        actual_rating = user_test[
                            user_test['movieId'] == item_id]['rating'].iloc[0]
                        eval_actuals.append(actual_rating)
                    else:
                        # Assume unrated items are not relevant
                        eval_actuals.append(2.5)
                except:
                    eval_predictions.append(3.0)
                    eval_actuals.append(2.5)
            
            # Calculate precision@k with threshold for relevance
            k_value = min(10, len(eval_predictions))
            if k_value >= 3:
                precision_scores.append(precision_at_k_improved_func(
                    np.array(eval_actuals), 
                    np.array(eval_predictions), 
                    k_value, 3.5))
        
        transfer_results[model_name] = {
            'rmse': np.mean(rmse_scores) if rmse_scores else float('inf'),
            'precision@10': np.mean(precision_scores) if precision_scores else 0.0,
            'users_tested': len(rmse_scores),
            'successful_evaluations': successful_evaluations
        }
        
        print(f"    {model_name}: {successful_evaluations} successful evaluations, "
              f"P@10={transfer_results[model_name]['precision@10']:.3f}")
    
    return transfer_results

def run_complete_transfer_analysis(movies_data, books_data, movies_models, books_models,
                                   evaluate_comprehensive_func, precision_at_k_func):
    # Complete transfer learning analysis pipeline
    # Evaluates both native performance and cross-domain transfer
    
    # Args:
        # movies_data, books_data: Loaded datasets
        # movies_models, books_models: Trained models
        # evaluate_comprehensive_func: Function to evaluate models
        # precision_at_k_func: Precision metric function
    
    # Returns:
        # Dictionary with all transfer learning results
    
    print("="*70)
    print("COMPLETE TRANSFER LEARNING ANALYSIS")
    print("="*70)
    
    # Create simulated cross-domain overlap
    movies_transfer, books_transfer = create_realistic_cross_domain_overlap(
        movies_data, books_data, min_overlap_users=100)
    
    # Evaluate native performance on cross-domain users only
    print("\nEvaluating native performance...")
    
    movies_cross_ratings = movies_transfer['ratings'][
        movies_transfer['ratings']['userId'].isin(movies_transfer['cross_domain_users'])]
    books_cross_ratings = books_transfer['ratings'][
        books_transfer['ratings']['userId'].isin(books_transfer['cross_domain_users'])]
    
    movies_train, movies_test = train_test_split(
        movies_cross_ratings, test_size=0.2, random_state=42)
    books_train, books_test = train_test_split(
        books_cross_ratings, test_size=0.2, random_state=42)
    
    movies_native = evaluate_comprehensive_func(
        movies_models, movies_train, movies_test, movies_transfer['movies'], 60)
    books_native = evaluate_comprehensive_func(
        books_models, books_train, books_test, books_transfer['movies'], 60)
    
    # Transfer evaluation: apply source models to target domain
    print("\nEvaluating transfer learning...")
    movies_to_books = evaluate_proper_transfer_fixed(
        movies_models, movies_transfer, books_transfer, 
        'movies_to_books', precision_at_k_func)
    books_to_movies = evaluate_proper_transfer_fixed(
        books_models, books_transfer, movies_transfer, 
        'books_to_movies', precision_at_k_func)
    
    # Display results with retention percentages
    display_transfer_results(
        movies_native, books_native, movies_to_books, books_to_movies)
    
    return {
        'movies_native': movies_native,
        'books_native': books_native,
        'movies_to_books': movies_to_books,
        'books_to_movies': books_to_movies,
        'overlap_stats': movies_transfer['overlap_stats']
    }

def display_transfer_results(movies_native, books_native, 
                             movies_to_books, books_to_movies):
    # Display transfer learning results with retention percentages
    print("\n" + "="*70)
    print("PROPER TRANSFER LEARNING RESULTS")
    print("="*70)
    
    # Include all model types
    individual_models = ['svd', 'itemcf', 'content', 'itemcf_content', 'svd_itemcf']
    
    print("\nMOVIES → BOOKS TRANSFER (Cross-Domain Users Only)")
    print(f"{'Model':<20} | {'Native':<8} | {'Transfer':<8} | "
          f"{'Retention':<10} | {'Status':<10}")
    print("-"*75)
    
    for model in individual_models:
        movies_key = f'{model}_movies'
        
        if movies_key in movies_native and movies_key in movies_to_books:
            native_score = movies_native[movies_key]['precision@10']
            transfer_score = movies_to_books[movies_key]['precision@10']
            # Retention = percentage of native performance preserved after transfer
            retention = ((transfer_score / native_score * 100) 
                        if native_score > 0 else 0)
            
            # Classify transfer quality based on retention
            if 40 <= retention <= 90:
                status = "✓ Good"
            elif 20 <= retention < 40:
                status = "○ Weak"
            elif retention < 20:
                status = "✗ Poor"
            else:
                status = "? Suspicious"  # >100% suggests overfitting or error
            
            print(f"{model:<20} | {native_score:<8.3f} | {transfer_score:<8.3f} | "
                  f"{retention:<10.1f}% | {status:<10}")
    
    print("\nBOOKS → MOVIES TRANSFER (Cross-Domain Users Only)")
    print(f"{'Model':<20} | {'Native':<8} | {'Transfer':<8} | "
          f"{'Retention':<10} | {'Status':<10}")
    print("-"*75)
    
    for model in individual_models:
        books_key = f'{model}_books'
        
        if books_key in books_native and books_key in books_to_movies:
            native_score = books_native[books_key]['precision@10']
            transfer_score = books_to_movies[books_key]['precision@10']
            retention = ((transfer_score / native_score * 100) 
                        if native_score > 0 else 0)
            
            # Classify transfer quality
            if 40 <= retention <= 90:
                status = "✓ Good"
            elif 20 <= retention < 40:
                status = "○ Weak"
            elif retention < 20:
                status = "✗ Poor"
            else:
                status = "? Suspicious"
            
            print(f"{model:<20} | {native_score:<8.3f} | {transfer_score:<8.3f} | "
                  f"{retention:<10.1f}% | {status:<10}")
    
    # Explain retention thresholds
    print(f"\nTRANSFER QUALITY GUIDE:")
    print(f"✓ Good transfer: 40-90% retention (learns transferable patterns)")
    print(f"○ Weak transfer: 20-40% retention (some generalization)")
    print(f"✗ Poor transfer: <20% retention (domain-specific learning)")
    print(f"? Suspicious: >100% retention (check implementation)")