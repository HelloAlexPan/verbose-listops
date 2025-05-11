def generate_single_sample(sample_index: int) -> dict | None:
    """Generate a single sample including AST, world data, narrative, and all metadata."""
    try:
        # Initialize the AST for this sample
        try:
            node = create_random_ast(
                max_depth=config.MAX_OPS,
                min_arity=config.MIN_ARITY,
                max_arity=config.MAX_BRANCH,
                min_atom_val=config.MIN_ATOM_VAL,
                max_atom_val=config.MAX_ATOM_VAL,
                early_term_prob=config.EARLY_TERMINATION_PROBABILITY,
            )
            logger.info(
                f"Generated AST with {node.num_ops} operations for sample {sample_index+1}"
            )
        except Exception as e:
            logger.error(f"AST creation failed for sample {sample_index+1}: {str(e)}")
            return None

        # Generate world data for this sample
        try:
            world_data = generate_world(
                node=node,
                sample_index=sample_index,
                min_chars=config.MIN_WORLD_CHARS,
                max_chars=config.MAX_WORLD_CHARS,
                num_concepts=random.randint(
                    config.MIN_WORLD_CONCEPTS, config.MAX_WORLD_CONCEPTS
                ),
            )
            logger.info(f"Generated world data for sample {sample_index+1}")
        except Exception as e:
            logger.error(
                f"World generation failed for sample {sample_index+1}: {str(e)}"
            )
            return None

        # Generate the narrative from the AST using the world data
        try:
            # Create a context object with the full sample's configs and data
            context = GenerationContext(
                config=config,
                beat_counter={"current": 1, "total": node.num_ops},
                tokens_used=0,
                allocated_tokens=config.MAX_TOTAL_TOKENS,
                scenes=[""],
                world_info=world_data,
                sample_index=sample_index,
            )

            # Generate the narrative
            narrative = generate_narrative(
                node=node,
                context=context,
                is_root=True,
                overall_ground_truth_answer=node.value,
            )
            logger.info(
                f"Generated narrative for sample {sample_index+1}: {context.tokens_used} tokens used"
            )
        except BeatGenerationError as e:
            logger.error(
                f"Narrative generation failed for sample {sample_index+1}: {str(e)}"
            )
            return None
        except Exception as e:
            logger.exception(
                f"Unexpected error during narrative generation for sample {sample_index+1}: {str(e)}"
            )
            return None

        # Prepare the sample data
        try:
            sample_data = {
                "id": sample_index,
                "ground_truth": node.value,
                "ast": node.to_dict(),
                "prompt": context.scenes[0],  # The full narrative
                "tokens_used": context.tokens_used,
                "world_data": world_data,
                # Add metadata about the experiment
                "metadata": {
                    "model_used": config.MODEL,
                    "config": {
                        "max_ops": config.MAX_OPS,
                        "max_branch": config.MAX_BRANCH,
                        "min_arity": config.MIN_ARITY,
                        "min_atom_val": config.MIN_ATOM_VAL,
                        "max_atom_val": config.MAX_ATOM_VAL,
                        "max_total_tokens": config.MAX_TOTAL_TOKENS,
                        "early_termination_probability": config.EARLY_TERMINATION_PROBABILITY,
                        "padding_max_tok_percent": config.PADDING_MAX_TOK_PERCENT,
                    },
                    "generation_timestamp": datetime.now().isoformat(),
                },
            }
            logger.info(f"Prepared sample data for sample {sample_index+1}")
            return sample_data
        except Exception as e:
            logger.error(
                f"Sample data preparation failed for sample {sample_index+1}: {str(e)}"
            )
            return None

    except Exception as e:
        logger.exception(
            f"Unexpected error in generate_single_sample for sample {sample_index+1}: {str(e)}"
        )
        return None
