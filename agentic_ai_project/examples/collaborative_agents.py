"""Collaborative Agents Example.

Demonstrates multiple agents working together, communicating,
and reaching consensus on decisions.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import CollaborativeAgent
from src.agents.collaborative_agent import CollaborativeConfig, MessageType
from src.utils import AgentLogger, MetricsCollector


def run_collaborative_demo():
    """Run a collaborative agents demonstration."""
    print("=" * 60)
    print("Collaborative Agents Demo")
    print("=" * 60)

    # Clear any previous agents
    CollaborativeAgent.clear_registry()

    # Create a team of collaborative agents
    agents = [
        CollaborativeAgent(
            name="Leader",
            config=CollaborativeConfig(consensus_threshold=0.6),
        ),
        CollaborativeAgent(
            name="Analyst",
            config=CollaborativeConfig(consensus_threshold=0.6),
        ),
        CollaborativeAgent(
            name="Executor",
            config=CollaborativeConfig(consensus_threshold=0.6),
        ),
        CollaborativeAgent(
            name="Validator",
            config=CollaborativeConfig(consensus_threshold=0.6),
        ),
    ]

    # Set up loggers
    loggers = {agent.name: AgentLogger(agent.name) for agent in agents}

    # Form the team - each agent knows about others
    print("\nForming team...")
    for agent in agents:
        for other in agents:
            if agent.id != other.id:
                agent.join_team(other.id)
        print(f"  {agent.name} joined team (ID: {agent.id[:8]}...)")

    print(f"\nTeam formed with {len(agents)} agents")
    print("-" * 60)

    # Simulate collaborative decision making
    print("\n### Phase 1: Information Sharing ###")

    # Leader broadcasts a task
    leader = agents[0]
    leader.send_message(
        recipient=None,  # Broadcast
        message_type=MessageType.BROADCAST,
        content={"task": "Analyze market data", "priority": "high"},
    )
    loggers["Leader"].info("Broadcasted task to team")
    print("Leader: Broadcasted task to team")

    # Process messages
    for agent in agents[1:]:
        messages = agent._process_inbox()
        for msg in messages:
            print(f"{agent.name}: Received message from {msg.sender[:8]}")
            loggers[agent.name].info(f"Received {msg.message_type.value} message")

    print("\n### Phase 2: Proposal and Voting ###")

    # Leader proposes a plan
    proposal = {
        "plan": "Execute market analysis using parallel processing",
        "resources_needed": ["data_source", "compute_cluster"],
        "estimated_time": "2 hours",
    }

    proposal_id = leader.propose(proposal)
    loggers["Leader"].info(f"Created proposal: {proposal_id[:8]}")
    print(f"Leader: Proposed plan (ID: {proposal_id[:8]})")

    # Other agents process proposals and vote
    for agent in agents[1:]:
        # Process incoming proposal
        messages = agent._process_inbox()
        for msg in messages:
            if msg.message_type == MessageType.PROPOSAL:
                # Simulate evaluation
                approve = agent.name != "Validator"  # Validator disagrees for demo
                reason = "Plan looks feasible" if approve else "Need more details"

                agent.vote(proposal_id, approve, reason)
                print(f"{agent.name}: Voted {'YES' if approve else 'NO'} - {reason}")
                loggers[agent.name].info(f"Voted {approve} on proposal")

    # Leader processes votes
    print("\n### Phase 3: Consensus Check ###")

    # Process incoming votes
    for _ in range(3):  # Process vote messages
        leader._process_inbox()
        for msg in leader.inbox:
            leader._handle_message(msg)
        leader.inbox.clear()

    # Check consensus
    consensus, ratio = leader.check_consensus(proposal_id)
    print(f"Leader: Checking consensus...")
    print(f"  Approval ratio: {ratio:.0%}")
    print(f"  Threshold: {leader.config.consensus_threshold:.0%}")
    print(f"  Consensus reached: {consensus}")

    loggers["Leader"].info(
        f"Consensus check: {consensus}",
        approval_ratio=ratio,
    )

    # Announce result
    print("\n### Phase 4: Execution ###")

    if consensus:
        print("Leader: Consensus reached! Proceeding with plan.")

        # Send execution instructions
        for agent in agents[1:]:
            leader.send_message(
                recipient=agent.id,
                message_type=MessageType.REQUEST,
                content={"action": "execute", "role": agent.name.lower()},
            )

        # Each agent acknowledges
        for agent in agents[1:]:
            messages = agent._process_inbox()
            for msg in messages:
                if msg.message_type == MessageType.REQUEST:
                    print(f"{agent.name}: Acknowledged execution task")
    else:
        print("Leader: Consensus not reached. Revising proposal...")

    # Summary
    print("\n" + "=" * 60)
    print("Collaboration Summary")
    print("=" * 60)

    print(f"\nTeam size: {len(agents)}")
    print(f"Messages exchanged: {sum(len(a.outbox) for a in agents)}")
    print(f"Proposals created: {len(leader.pending_proposals)}")
    print(f"Final consensus: {consensus} ({ratio:.0%} approval)")

    # Print agent summaries
    print("\nAgent Activity:")
    for agent in agents:
        state = agent.get_state_summary()
        print(f"  {agent.name}:")
        print(f"    Actions: {state['action_count']}")
        print(f"    Messages sent: {len(agent.outbox)}")

    # Cleanup
    CollaborativeAgent.clear_registry()

    return consensus


def demo_conflict_resolution():
    """Demonstrate conflict resolution between agents."""
    print("\n" + "=" * 60)
    print("Conflict Resolution Demo")
    print("=" * 60)

    CollaborativeAgent.clear_registry()

    # Create agents with conflicting views
    agent_a = CollaborativeAgent(name="Optimist")
    agent_b = CollaborativeAgent(name="Pessimist")
    agent_c = CollaborativeAgent(name="Mediator")

    # Form network
    for agent in [agent_a, agent_b, agent_c]:
        for other in [agent_a, agent_b, agent_c]:
            if agent.id != other.id:
                agent.join_team(other.id)

    print("\nScenario: Two agents disagree, mediator helps resolve")

    # Optimist proposes aggressive plan
    proposal_id = agent_a.propose({
        "strategy": "aggressive_expansion",
        "risk_level": "high",
    })
    print(f"Optimist: Proposed aggressive expansion")

    # Process and vote
    for agent in [agent_b, agent_c]:
```suggestion
        messages = agent.step(None)
        if messages and isinstance(messages, list):
            for msg in messages:
                agent._handle_message(msg)

    # Check result
    consensus, ratio = agent_a.check_consensus(proposal_id)
    print(f"Result: Consensus={consensus}, Approval={ratio:.0%}")

    if not consensus:
        print("\nMediator: Let's find middle ground...")
        # Mediator proposes compromise
        compromise_id = agent_c.propose({
            "strategy": "moderate_expansion",
            "risk_level": "medium",
        })

        # Both vote on compromise
        for agent in [agent_a, agent_b]:
            agent._process_inbox()
            for msg in agent.inbox:
                agent._handle_message(msg)
            agent.inbox.clear()
            agent.vote(compromise_id, True, "Acceptable compromise")

        consensus, ratio = agent_c.check_consensus(compromise_id)
        print(f"Compromise result: Consensus={consensus}, Approval={ratio:.0%}")

    CollaborativeAgent.clear_registry()


if __name__ == "__main__":
    run_collaborative_demo()
    demo_conflict_resolution()
