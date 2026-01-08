import { NextResponse } from 'next/server';
import { getRoutingStats } from '@/lib/model-router';

export async function GET() {
  try {
    const stats = getRoutingStats();
    
    return NextResponse.json({
      success: true,
      data: stats,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Error fetching routing stats:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch routing statistics',
      },
      { status: 500 }
    );
  }
}

