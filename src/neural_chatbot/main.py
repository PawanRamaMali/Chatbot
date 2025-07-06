#!/usr/bin/env python3
"""
Main entry point for Neural Chatbot CLI
"""

import sys
import argparse
from pathlib import Path

from .config.settings import Settings
from .core.chatbot import NeuralChatbot
from .core.trainer import ModelTrainer
from .utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Neural Chatbot - An intelligent AI chatbot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neural-chatbot chat                    # Start interactive chat
  neural-chatbot train                   # Train the model
  neural-chatbot train --plot            # Train with visualization
  neural-chatbot serve                   # Start web server
  neural-chatbot status                  # Show chatbot status
        """
    )
    
    # Global arguments
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--version', action='version', version='Neural Chatbot 1.0.0')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start interactive chat')
    chat_parser.add_argument('--user-id', type=str, help='User ID for conversation tracking')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the chatbot model')
    train_parser.add_argument('--intents', type=str, help='Path to intents file')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--plot', action='store_true', help='Generate training plots')
    train_parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start web server')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    serve_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    serve_parser.add_argument('--workers', type=int, default=1, help='Number of workers (production)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show chatbot status')
    status_parser.add_argument('--detailed', action='store_true', help='Show detailed statistics')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--save-report', action='store_true', help='Save evaluation report')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model or data')
    export_parser.add_argument('--type', choices=['model', 'conversations', 'config'], 
                              required=True, help='What to export')
    export_parser.add_argument('--output', type=str, required=True, help='Output file path')
    export_parser.add_argument('--format', choices=['json', 'h5', 'tflite'], 
                              default='json', help='Export format')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level)
    
    # Load settings
    try:
        settings = Settings(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Execute command
    if args.command == 'chat':
        run_chat(settings, args)
    elif args.command == 'train':
        run_training(settings, args)
    elif args.command == 'serve':
        run_server(settings, args)
    elif args.command == 'status':
        show_status(settings, args)
    elif args.command == 'evaluate':
        run_evaluation(settings, args)
    elif args.command == 'export':
        run_export(settings, args)
    else:
        parser.print_help()
        sys.exit(1)


def run_chat(settings: Settings, args):
    """Run interactive chat"""
    logger.info("Starting interactive chat...")
    
    try:
        chatbot = NeuralChatbot(settings)
        chatbot.initialize()
        chatbot.chat_interactive()
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        sys.exit(1)


def run_training(settings: Settings, args):
    """Run model training"""
    logger.info("Starting model training...")
    
    try:
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
        
        print(f"\n✅ Training completed successfully!")
        print(f"Final accuracy: {results['final_accuracy']:.4f}")
        print(f"Validation accuracy: {results['validation_accuracy']:.4f}")
        
        # Run evaluation if requested
        if args.evaluate:
            print("\n🔍 Running evaluation...")
            eval_results = trainer.evaluate_model_performance()
            print("✅ Evaluation completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def run_server(settings: Settings, args):
    """Run web server"""
    logger.info("Starting web server...")
    
    try:
        # Override settings
        if args.host:
            settings.api.host = args.host
        if args.port:
            settings.api.port = args.port
        if args.debug:
            settings.api.debug = args.debug
        
        # Import and run app
        from .api.app import create_app
        
        app = create_app()
        app.run(
            host=settings.api.host,
            port=settings.api.port,
            debug=settings.api.debug,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


def show_status(settings: Settings, args):
    """Show chatbot status"""
    try:
        chatbot = NeuralChatbot(settings)
        chatbot.initialize()
        
        # Get status information
        health = chatbot.health_check()
        stats = chatbot.get_statistics()
        
        print("=" * 60)
        print("NEURAL CHATBOT STATUS")
        print("=" * 60)
        
        # Health status
        print(f"Status: {health['status']}")
        print(f"Timestamp: {health['timestamp']}")
        
        # Basic info
        if stats['status']['initialized']:
            print(f"Initialized: ✅")
            print(f"Trained: {'✅' if stats['status']['trained'] else '❌'}")
            print(f"Model Loaded: {'✅' if stats['status']['model_loaded'] else '❌'}")
        else:
            print(f"Initialized: ❌")
        
        if args.detailed and stats['status']['initialized']:
            print("\nDETAILED STATISTICS:")
            print("-" * 20)
            print(f"Vocabulary Size: {stats['settings']['vocabulary_size']}")
            print(f"Number of Intents: {stats['settings']['num_intents']}")
            print(f"Confidence Threshold: {stats['settings']['confidence_threshold']}")
            
            if stats['conversations']['total_users'] > 0:
                print(f"\nCONVERSATION STATISTICS:")
                print("-" * 25)
                print(f"Total Users: {stats['conversations']['total_users']}")
                print(f"Total Interactions: {stats['conversations']['total_interactions']}")
                print(f"Average Interactions per User: {stats['conversations']['average_interactions_per_user']:.1f}")
                
                if stats['conversations']['most_common_intents']:
                    print(f"\nMost Common Intents:")
                    for intent, count in stats['conversations']['most_common_intents'][:3]:
                        print(f"  {intent}: {count}")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        sys.exit(1)


def run_evaluation(settings: Settings, args):
    """Run model evaluation"""
    try:
        chatbot = NeuralChatbot(settings)
        chatbot.initialize()
        
        if not chatbot.is_initialized:
            print("❌ Chatbot not initialized. Please train the model first.")
            sys.exit(1)
        
        print("🔍 Running model evaluation...")
        
        # Run evaluation
        results = chatbot.evaluate_model()
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        
        # Show basic metrics
        for metric, value in results.items():
            if isinstance(value, (int, float)) and metric != 'confusion_matrix':
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Show classification report
        if 'classification_report' in results:
            print(f"\nClassification Report:")
            print("-" * 25)
            report = results['classification_report']
            
            for intent, metrics in report.items():
                if isinstance(metrics, dict) and intent not in ['accuracy', 'macro avg', 'weighted avg']:
                    print(f"{intent:15} - Precision: {metrics.get('precision', 0):.3f}, "
                          f"Recall: {metrics.get('recall', 0):.3f}, "
                          f"F1: {metrics.get('f1-score', 0):.3f}")
        
        # Save report if requested
        if args.save_report:
            from datetime import datetime
            import json
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"reports/evaluation_report_{timestamp}.json"
            
            Path("reports").mkdir(exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n📄 Evaluation report saved to: {report_path}")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


def run_export(settings: Settings, args):
    """Export model or data"""
    try:
        chatbot = NeuralChatbot(settings)
        
        if args.type in ['model', 'conversations']:
            chatbot.initialize()
        
        print(f"📦 Exporting {args.type} to {args.output}...")
        
        if args.type == 'model':
            if not chatbot.is_initialized:
                print("❌ No trained model to export.")
                sys.exit(1)
            
            if args.format == 'h5':
                chatbot.model.save_model(args.output)
            elif args.format == 'tflite':
                chatbot.model.export_model(args.output, 'tflite')
            else:  # json
                model_info = {
                    'model_summary': chatbot.model.get_model_summary(),
                    'vocabulary': chatbot.processor.words,
                    'classes': chatbot.processor.classes,
                    'settings': settings.to_dict(),
                    'export_timestamp': datetime.now().isoformat()
                }
                
                import json
                with open(args.output, 'w') as f:
                    json.dump(model_info, f, indent=2)
        
        elif args.type == 'conversations':
            chatbot.export_conversation_history(args.output)
        
        elif args.type == 'config':
            settings.save_to_file(args.output)
        
        print(f"✅ Export completed: {args.output}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()