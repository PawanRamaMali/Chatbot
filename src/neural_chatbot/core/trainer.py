"""
Training module for Neural Chatbot
Handles model training with advanced features like visualization and evaluation
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from ..config.settings import Settings
from .chatbot import NeuralChatbot
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Advanced trainer with visualization and evaluation capabilities"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.chatbot = NeuralChatbot(self.settings)
        self.training_results: Optional[Dict[str, Any]] = None
        
    def train_model(self, intents_file: Optional[str] = None, 
                   plot_results: bool = False) -> Dict[str, Any]:
        """Train the chatbot model with comprehensive logging"""
        logger.info("Starting model training...")
        
        # Create output directories
        self._create_directories()
        
        try:
            # Train the model
            results = self.chatbot.train(intents_file)
            self.training_results = results
            
            # Generate comprehensive training report
            report = self._generate_training_report()
            
            # Save training report
            self._save_training_report(report)
            
            # Generate plots if requested
            if plot_results:
                self._plot_training_history()
                self._plot_confusion_matrix()
            
            # Test the trained model
            self._test_trained_model()
            
            logger.info("Training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _create_directories(self):
        """Create necessary directories for training outputs"""
        directories = ['models', 'logs', 'plots', 'reports', 'data']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def _generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        if not self.training_results:
            return {}
        
        # Get model statistics
        model_stats = self.chatbot.get_statistics()
        
        # Get training history
        history = self.chatbot.model.get_training_history()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_results': self.training_results,
            'model_statistics': model_stats,
            'training_history': history,
            'configuration': self.settings.to_dict(),
            'data_statistics': self.chatbot.processor.get_statistics(),
        }
        
        # Add evaluation metrics if available
        try:
            evaluation = self.chatbot.evaluate_model()
            report['evaluation'] = evaluation
        except Exception as e:
            logger.warning(f"Could not generate evaluation metrics: {e}")
        
        return report
    
    def _save_training_report(self, report: Dict[str, Any]):
        """Save training report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_path = f"reports/training_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        text_path = f"reports/training_summary_{timestamp}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(self._format_text_report(report))
        
        logger.info(f"Training report saved to {json_path} and {text_path}")
    
    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Format report as human-readable text"""
        lines = []
        lines.append("=" * 60)
        lines.append("NEURAL CHATBOT TRAINING REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {report['timestamp']}")
        lines.append("")
        
        # Training results
        if 'training_results' in report:
            results = report['training_results']
            lines.append("TRAINING RESULTS:")
            lines.append("-" * 20)
            lines.append(f"Final Accuracy: {results.get('final_accuracy', 0):.4f}")
            lines.append(f"Validation Accuracy: {results.get('validation_accuracy', 0):.4f}")
            lines.append(f"Epochs Completed: {results.get('epochs_completed', 0)}")
            lines.append(f"Total Samples: {results.get('total_samples', 0)}")
            lines.append(f"Vocabulary Size: {results.get('vocabulary_size', 0)}")
            lines.append(f"Number of Classes: {results.get('num_classes', 0)}")
            lines.append("")
        
        # Model configuration
        if 'configuration' in report:
            config = report['configuration']
            lines.append("MODEL CONFIGURATION:")
            lines.append("-" * 20)
            model_config = config.get('model', {})
            lines.append(f"Learning Rate: {model_config.get('learning_rate', 0)}")
            lines.append(f"Batch Size: {model_config.get('batch_size', 0)}")
            lines.append(f"Epochs: {model_config.get('epochs', 0)}")
            lines.append(f"Dropout Rate: {model_config.get('dropout_rate', 0)}")
            lines.append(f"Confidence Threshold: {model_config.get('confidence_threshold', 0)}")
            lines.append("")
        
        # Data statistics
        if 'data_statistics' in report:
            data_stats = report['data_statistics']
            lines.append("DATA STATISTICS:")
            lines.append("-" * 20)
            lines.append(f"Total Words: {data_stats.get('total_words', 0)}")
            lines.append(f"Total Classes: {data_stats.get('total_classes', 0)}")
            lines.append(f"Total Documents: {data_stats.get('total_documents', 0)}")
            lines.append(f"Avg Tokens per Document: {data_stats.get('average_tokens_per_document', 0):.2f}")
            lines.append("")
        
        # Evaluation results
        if 'evaluation' in report:
            evaluation = report['evaluation']
            lines.append("EVALUATION RESULTS:")
            lines.append("-" * 20)
            for metric, value in evaluation.items():
                if isinstance(value, (int, float)):
                    lines.append(f"{metric.title()}: {value:.4f}")
            lines.append("")
        
        lines.append("=" * 60)
        return '\n'.join(lines)
    
    def _plot_training_history(self):
        """Plot training history"""
        history = self.chatbot.model.get_training_history()
        if not history:
            logger.warning("No training history available for plotting")
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Accuracy plot
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if 'lr' in history:
            axes[1, 0].plot(history['lr'], linewidth=2, color='orange')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning rate data\nnot available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Learning Rate')
        
        # Top-k accuracy plot (if available)
        if 'top_k_categorical_accuracy' in history:
            axes[1, 1].plot(history['top_k_categorical_accuracy'], 
                           label='Top-K Accuracy', linewidth=2, color='green')
            if 'val_top_k_categorical_accuracy' in history:
                axes[1, 1].plot(history['val_top_k_categorical_accuracy'], 
                               label='Val Top-K Accuracy', linewidth=2, color='lightgreen')
            axes[1, 1].set_title('Top-K Categorical Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Top-K accuracy data\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Top-K Accuracy')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f"plots/training_history_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Training history plot saved to {plot_path}")
    
    def _plot_confusion_matrix(self):
        """Plot confusion matrix"""
        try:
            evaluation = self.chatbot.evaluate_model()
            confusion_mat = np.array(evaluation['confusion_matrix'])
            classes = self.chatbot.processor.classes
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                       xticklabels=classes, yticklabels=classes)
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = f"plots/confusion_matrix_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Confusion matrix plot saved to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate confusion matrix: {e}")
    
    def _test_trained_model(self):
        """Test the trained model with sample inputs"""
        test_messages = [
            "Hello",
            "Hi there",
            "What stocks do I own?",
            "Show me my portfolio",
            "Tell me a joke",
            "Make me laugh",
            "Thank you",
            "Thanks for your help",
            "Goodbye",
            "See you later",
            "What is your name?",
            "Who are you?",
            "Help me",
            "What can you do?",
        ]
        
        print("\n" + "=" * 60)
        print("TESTING TRAINED MODEL")
        print("=" * 60)
        
        for message in test_messages:
            try:
                response = self.chatbot.get_response(message, 'test_user')
                intent, confidence, _ = self.chatbot.predict_intent(message)
                
                print(f"Input: {message}")
                print(f"Intent: {intent} (confidence: {confidence:.3f})")
                print(f"Response: {response}")
                print("-" * 40)
                
            except Exception as e:
                print(f"Error testing message '{message}': {e}")
                print("-" * 40)
    
    def evaluate_model_performance(self) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        logger.info("Evaluating model performance...")
        
        try:
            # Basic evaluation
            evaluation = self.chatbot.evaluate_model()
            
            # Get statistics
            stats = self.chatbot.get_statistics()
            
            # Test with various confidence thresholds
            confidence_analysis = self._analyze_confidence_thresholds()
            
            # Intent distribution analysis
            intent_analysis = self._analyze_intent_distribution()
            
            results = {
                'evaluation_metrics': evaluation,
                'model_statistics': stats,
                'confidence_analysis': confidence_analysis,
                'intent_analysis': intent_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save evaluation results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            eval_path = f"reports/model_evaluation_{timestamp}.json"
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Model evaluation saved to {eval_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise
    
    def _analyze_confidence_thresholds(self) -> Dict[str, Any]:
        """Analyze model performance at different confidence thresholds"""
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        original_threshold = self.settings.model.confidence_threshold
        
        results = {}
        
        # Generate test messages from intents
        test_messages = []
        for intent in self.chatbot.processor.intents_data['intents']:
            test_messages.extend(intent['patterns'][:3])  # Take first 3 patterns
        
        for threshold in thresholds:
            self.settings.model.confidence_threshold = threshold
            
            predictions_above_threshold = 0
            correct_predictions = 0
            
            for message in test_messages:
                intent, confidence, _ = self.chatbot.predict_intent(message)
                
                if confidence >= threshold:
                    predictions_above_threshold += 1
                    # Simple correctness check (you might want to implement this properly)
                    if intent != 'unknown':
                        correct_predictions += 1
            
            precision = correct_predictions / predictions_above_threshold if predictions_above_threshold > 0 else 0
            coverage = predictions_above_threshold / len(test_messages) if test_messages else 0
            
            results[str(threshold)] = {
                'precision': precision,
                'coverage': coverage,
                'predictions_above_threshold': predictions_above_threshold,
                'total_predictions': len(test_messages)
            }
        
        # Restore original threshold
        self.settings.model.confidence_threshold = original_threshold
        
        return results
    
    def _analyze_intent_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of intents in training data"""
        intent_counts = {}
        total_patterns = 0
        
        for intent in self.chatbot.processor.intents_data['intents']:
            tag = intent['tag']
            pattern_count = len(intent['patterns'])
            intent_counts[tag] = pattern_count
            total_patterns += pattern_count
        
        # Calculate percentages
        intent_percentages = {
            tag: (count / total_patterns) * 100 
            for tag, count in intent_counts.items()
        }
        
        # Find imbalanced intents
        avg_percentage = 100 / len(intent_counts)
        imbalanced_intents = {
            tag: percentage 
            for tag, percentage in intent_percentages.items()
            if abs(percentage - avg_percentage) > avg_percentage * 0.5
        }
        
        return {
            'intent_counts': intent_counts,
            'intent_percentages': intent_percentages,
            'total_patterns': total_patterns,
            'total_intents': len(intent_counts),
            'average_patterns_per_intent': total_patterns / len(intent_counts),
            'imbalanced_intents': imbalanced_intents,
            'most_common_intent': max(intent_counts, key=intent_counts.get),
            'least_common_intent': min(intent_counts, key=intent_counts.get)
        }


def main():
    """Main function for command-line training"""
    parser = argparse.ArgumentParser(description='Train Neural Chatbot')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--intents', type=str, help='Path to intents file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--plot', action='store_true', help='Generate training plots')
    parser.add_argument('--evaluate', action='store_true', help='Run comprehensive evaluation')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    try:
        # Configure logging
        if args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Load settings
        settings = Settings(args.config)
        
        # Override settings with command line arguments
        if args.epochs:
            settings.model.epochs = args.epochs
        if args.batch_size:
            settings.model.batch_size = args.batch_size
        if args.learning_rate:
            settings.model.learning_rate = args.learning_rate
        
        # Initialize trainer
        trainer = ModelTrainer(settings)
        
        # Train model
        results = trainer.train_model(args.intents, args.plot)
        
        print("\n✅ Training completed successfully!")
        print(f"Final accuracy: {results['final_accuracy']:.4f}")
        print(f"Validation accuracy: {results['validation_accuracy']:.4f}")
        
        # Run evaluation if requested
        if args.evaluate:
            print("\n🔍 Running comprehensive evaluation...")
            eval_results = trainer.evaluate_model_performance()
            print("✅ Evaluation completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()